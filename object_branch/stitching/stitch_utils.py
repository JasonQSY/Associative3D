import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
import random
import collections
import pickle
import pdb

code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(code_root, '..'))
from object_branch.renderer import utils as render_utils
from object_branch.utils import suncg_parse
from object_branch.utils.chamferdist.chamferdist import ChamferDistance

cal_chamfer_dist = ChamferDistance()

# global constants
# detection = True
#detection = False
max_object_classes = 10
max_rois = 100

trans_class_file = '../cachedir/kmeans_trans_cluster_centers_v3_60.npy'
rot_class_file = '../cachedir/kmeans_rots_cluster_centers_v3_30.npy'
tran_class_map = np.load(trans_class_file)
rot_class_map = np.load(rot_class_file)
trans_kmeans_file = '../cachedir/kmeans_trans_60.pkl'
rots_kmeans_file = '../cachedir/kmeans_rots_30.pkl'
assert(os.path.exists(trans_kmeans_file))
assert(os.path.exists(rots_kmeans_file))
kmeans_trans = joblib.load(trans_kmeans_file)
kmeans_rots = joblib.load(rots_kmeans_file)
#cal_chamfer_dist = ChamferDistance()


def class2tran(cls):
    assert((cls >= 0).all() and (cls < tran_class_map.shape[0]).all())
    return tran_class_map[cls]

def class2quat(cls):
    assert((cls >= 0).all() and (cls < rot_class_map.shape[0]).all())
    return rot_class_map[cls]

def xyz2class(x, y, z):
    return kmeans_trans.predict([[x,y,z]])

def quat2class(w, xi, yi, zi):
    return kmeans_rots.predict([[w, xi, yi, zi]])

def is_gt_tran_rot_in_search_space(gt_tran, gt_rot, tran_logits, rot_logits, topk_tran=10, topk_rot=5, ):
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    return xyz2class(*gt_tran) in tran_proposals, quat2class(*gt_rot) in rot_proposals

def volume_iou(pred, gt, thresh=0.5):
    pred = torch.FloatTensor(pred)
    gt = torch.FloatTensor(gt)
    gt = gt.float().ge(0.5)
    pred = pred.float().ge(thresh)
    intersection = torch.mul(gt, pred).sum()
    union = gt.sum() + pred.sum() - intersection
    intersection = intersection.item()
    union = union.item()
    if union < 1:
        return 0.
    return intersection/union

def chamfer_distance(pcd1, pcd2):
    # pcu implementation
    #M = pcu.pairwise_distances(pcd1, pcd2)
    #if len(M.shape) == 2:
    #    M = M[np.newaxis, :, :]
    #chamfer_dist = M.min(1).mean() + M.min(2).mean()

    # pytorch impl
    # pcd1 = torch.FloatTensor(pcd1[np.newaxis, :]).cuda().contiguous()
    # pcd2 = torch.FloatTensor(pcd2[np.newaxis, :]).cuda().contiguous()
    pcd1 = torch.FloatTensor(pcd1).cuda().contiguous()
    pcd2 = torch.FloatTensor(pcd2).cuda().contiguous()
    dist1, dist2, _, _ = cal_chamfer_dist(pcd1, pcd2)
    chamfer_dist = dist1.mean(axis=1) + dist2.mean(axis=1)
    #chamfer_dist = torch.clamp(chamfer_dist, max=400).cpu().numpy()
    chamfer_dist = torch.clamp(chamfer_dist, max=40).cpu().numpy()
    return chamfer_dist


def chamfer_metric(cd_pairs, metric="avg"):
    '''
    cd_pairs: # of matching pairs x # of transformation proposals
    return: index of best transformation proposals, and cost
    '''
    cd_pairs = np.array(cd_pairs)
    if metric == 'avg':
        cost = np.average(cd_pairs, axis=0)
    elif metric == 'log_avg':
        # log(x + 1) -> 0 ~ inf
        cost = np.average(np.log(cd_pairs+1), axis=0)
    else:
        raise NotImplementedError
    
    return cost
    #return np.argmin(cost), cost[np.argmin(cost)]
    #try:
    #    return np.argmin(cost), cost[np.argmin(cost)]
    #except:
    #    raise RuntimeError("return np.argmin(cost), cost[np.argmin(cost)]")


def merge_codes(obj, mobj, method='avg', tester=None):
    """
    Merge codes. obj will be modified.
    """
    if method == 'avg':
        # the standard approach. average trans and scale.
        # pick up one shape and rot at random.
        idx = int(random.random() > 0.5)
        obj['shape'] = [obj['shape'], mobj['shape']][idx]
        obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
        obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
        obj['quat'] = [obj['quat'], mobj['quat']][idx]
        obj['cmap'] = 'green'
    elif method == 'avg_voxel':
        # average voxel, trans and scale, pick up one rot at random.
        # NOTE: this is not expected to work.
        idx = int(random.random() > 0.5)
        obj['shape'] = obj['shape'] / 2 + mobj['shape'] / 2
        obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
        obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
        obj['quat'] = [obj['quat'], mobj['quat']][idx]
        obj['cmap'] = 'green'
    elif method == 'random':
        # pick up one obj at random.
        idx = int(random.random() > 0.5)
        obj['shape'] = [obj['shape'], mobj['shape']][idx]
        obj['trans'] = [obj['trans'], mobj['trans']][idx]
        obj['scale'] = [obj['scale'], mobj['scale']][idx]
        obj['quat'] = [obj['quat'], mobj['quat']][idx]
        obj['cmap'] = 'green'
    elif method == 'avg_rot':
        # average rot, trans and scale, pick up one shape at random.
        # NOTE: this is not expected to work.
        idx = int(random.random() > 0.5)
        obj['shape'] = [obj['shape'], mobj['shape']][idx]
        obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
        obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
        obj['quat'] = mean_quaternion(obj['quat'], mobj['quat'])
        obj['cmap'] = 'green'
    elif method == 'avg_shape':
        # average shape codes, trans and scale, pick up one rot at random.
        assert(tester is not None)
        idx = int(random.random() > 0.5)
        avg_shape_code = (obj['shape_code'] + mobj['shape_code']) / 2
        avg_shape_code = avg_shape_code[np.newaxis, :]
        avg_shape_code = torch.FloatTensor(avg_shape_code).cuda()
        avg_shape = tester.decode_shape(avg_shape_code)
        avg_shape = avg_shape[0,0].cpu().detach().numpy()
        obj['shape'] = avg_shape
        obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
        obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
        obj['quat'] = [obj['quat'], mobj['quat']][idx]
        obj['cmap'] = 'green'
    else:
        raise NotImplementedError

    if 'node_ids' in obj and 'node_ids' in mobj:
        assert(len(obj['node_ids']) == 1)
        assert(len(mobj['node_ids']) == 1)
        if obj['node_ids'][0] != mobj['node_ids'][0]:
            obj['node_ids'] = [obj['node_ids'][0], mobj['node_ids'][0]]
    if 'scores' in obj and 'scores' in mobj:
        assert(len(obj['scores']) == 1)
        assert(len(mobj['scores']) == 1)
        obj['scores'] = [obj['scores'][0], mobj['scores'][0]]

    return obj


def mean_quaternion(q1, q2):
    """
    The mean of quaternion q1 and q2 is the eigenvector corresponding to 
    the maximal eigenvalue of matrix [q1; q2] * [q1; q2]^T.
    Both q1 and q2 are 4d numpy.array.
    """
    s = np.concatenate((q1[:, np.newaxis], q2[:, np.newaxis]), axis=1)
    prod = s @ s.T
    evals, evecs = np.linalg.eig(prod)
    idx_max = evals.argmax()
    avg = evecs[:, idx_max]
    return avg


def forward_affinity(id1, id2, mesh_dir=None):
    """
    Calculate affinity matrix according to id1 and id2.
    Save the affinity matrix to mesh_dir if it is not None.
    """
    valid_bs = 1
    affinity_pred = torch.zeros([max_rois, max_rois], dtype=torch.float32)
    if id1.shape[0] == 0 or id2.shape[0] == 0:
        affinity_pred = affinity_pred.numpy()
        if mesh_dir is not None:
            plt.figure()
            sns.heatmap(affinity_pred, vmin=0.0, vmax=1.0)
            plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
            plt.close()
        return affinity_pred
    affinity_pred = affinity_pred.cuda()
    objs_in_view1 = id1
    objs_in_view2 = id2
    
    norms = (torch.norm(objs_in_view1, p=2, dim=1) + 1e-3).detach()
    objs_in_view1 = objs_in_view1.div(norms.view(-1,1).expand_as(objs_in_view1))
    norms = (torch.norm(objs_in_view2, p=2, dim=1) + 1e-3).detach()
    objs_in_view2 = objs_in_view2.div(norms.view(-1,1).expand_as(objs_in_view2))
    prod = torch.mm(
        objs_in_view1, torch.t(objs_in_view2)
    ) * 5
    prod = torch.sigmoid(prod)
    sz_i = prod.shape[0]
    sz_j = prod.shape[1]
    #if sz_i > 20 or sz_j > 20:
    #shutil.rmtree(mesh_dir)
    #return None
    affinity_pred[:sz_i, :sz_j] = prod
        
    #match = affinity_pred.argmax(dim=1).cpu().numpy()
    affinity_pred = affinity_pred.cpu().detach().numpy()

    # visualization
    if mesh_dir is not None:
        max_sz = max(sz_i, sz_j)
        if max_sz < 5:
            max_sz = 5
        elif max_sz < 10:
            max_sz = 10
        affinity_vis = affinity_pred[:max_sz, :max_sz]
        plt.figure()
        sns.heatmap(affinity_vis, annot=True, vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
        plt.close()

    return affinity_pred


def get_matching_list(affinity_m, affinity_idx, sz_i):
    matching_list = []
    for i in range(sz_i):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= 0.5:
                continue
            options.append(j)
        if len(options) == 0:
            options.append(-1)
        matching_list.append(options)
    return matching_list


def clean_checkpoint_file(ckpt_file):
    checkpoint = torch.load(ckpt_file)
    keys = checkpoint.keys()
    
    temp = [key for key in keys if 'relative_quat_predictor' in key ] +  [key for key in keys if 'relative_encoder.encoder_joint_scale' in key]
    if len(temp) > 0:
        for t in temp:
            checkpoint.pop(t)

        torch.save(checkpoint, ckpt_file)


def save_results(mesh_dir, codes, codes_gt, rel_pose, 
                 rel_tran_after_stitch, rel_rot_after_stitch, 
                 affinity_pred, affinity_gt, matching_pred, prefix='results'):
    results = {
        'mesh_dir': mesh_dir,
        'objects_pred': codes,
        'objects_gt': codes_gt,
        'relpose': rel_pose,
        'relpose_after_stitch': {
            'rotation': rel_rot_after_stitch,
            'translation': rel_tran_after_stitch,
        },
        'affinity_pred': affinity_pred,
        'affinity_gt': affinity_gt,
        'matching_pred': matching_pred,
    }
    f = open(os.path.join(mesh_dir, prefix + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()


def save_codes_mesh(mesh_dir, code_list, prefix='codes'):
    mesh_file=os.path.join(mesh_dir, prefix + '.obj')
    new_codes_list = suncg_parse.convert_codes_list_to_old_format(code_list)
    render_utils.save_parse(mesh_file, new_codes_list, save_objectwise=False, thresh=0.1)
    

def convert_codes_list_to_old_format(codes_list):
    new_codes_list = []
    ## 0 --> shape
    ## 1 --> scale
    ## 2 --> quat
    ## 3 --> trans
    ## 4 --> cmap
    for code in codes_list:
        new_code = ()
        new_code  += (code['shape_edge'], code['scale'], )
        if code['quat'].ndim > 1:  ## to handle symmetry stuff.
            new_code += (code['quat'][0], )
        else:
            new_code += (code['quat']  ,)
        new_code += (code['trans'] ,)
        if 'cmap' in code:
            new_code += (code['cmap'], )
        new_codes_list.append(new_code)
    return new_codes_list


def object_to_pcd(obj, edge=True):
    """
    Covert an entry in code_list to point cloud.
    """
    if 'shape_edge' in obj and edge:
        temp = convert_codes_list_to_old_format([obj])
    else:
        temp = suncg_parse.convert_codes_list_to_old_format([obj])
    volume, transform = render_utils.prediction_to_entity(temp[0])
    points = render_utils.voxels_to_points(volume)
    points = points / 32 - 0.5
    n_verts = points.shape[0]
    v_homographic = np.concatenate((points, np.ones((n_verts, 1))), axis=1).transpose()
    v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()
    return v_transformed


def pcd_to_obj_save(pcds, fname='debug.obj'):
    with open(fname,'w') as f:
        for pcd in pcds:
            f.write("v {} {} {}\n".format(pcd[0], pcd[1], pcd[2]))


def save_affinity_after_stitch(affinity_pred, sz_i, sz_j, matching, mesh_dir):
    try:
        max_sz = max(sz_i, sz_j)
        if max_sz < 5:
            max_sz = 5
        elif max_sz < 10:
            max_sz = 10
        text = np.array([['']*sz_j]*sz_i)
        for i,j in enumerate(matching):
            if j != -1:
                text[i][j] = '*'
        affinity_vis = affinity_pred[:max_sz, :max_sz]
        labels = (np.asarray(["{}\n{:.2f}".format(text,data) for text, data in zip(text.flatten(), affinity_pred[:sz_i, :sz_j].flatten())])).reshape(text.shape)
        labels_full = np.array([['']*max_sz]*max_sz).astype('<U6')
        labels_full[:sz_i,:sz_j] = labels
        plt.figure()
        sns.heatmap(affinity_vis, annot=labels_full, fmt='s', vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
        plt.close()
    except:
        plt.figure()
        sns.heatmap(affinity_pred[:max_sz, :max_sz], vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
        plt.close()
        pass


def is_valid_matching(matching):
    """
    Matching proposal should not contain duplicate values except -1.
    """
    cnt = collections.Counter(matching)
    for k in cnt:
        if k != -1 and cnt[k] > 1:
            return False
    return True
