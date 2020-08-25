import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os.path as osp
import os
import scipy.misc
import scipy.io as sio
import torch
import imageio
import pdb
from absl import app, flags
import cv2
from tqdm import tqdm
import random
import shutil
#from pyquaternion import Quaternion
import pickle
import itertools
from copy import deepcopy
from PIL import Image
import math
from time import time
import multiprocessing
from multiprocessing import Pool, Process, Queue


code_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(code_root, '..'))
from object_branch.stitching import siamese_demo_utils as demo_utils
from object_branch.renderer import utils as render_utils
from object_branch.utils import suncg_parse

from stitch_methods import *
from stitch_utils import *

torch.cuda.empty_cache()

# flags
flags.FLAGS(['demo'])
flags.DEFINE_string('output_dir', '/x/syqian/mesh_output', 'output_dir')
flags.DEFINE_string('suncg_dir', '/x/syqian/suncg', 'suncg dir')
flags.DEFINE_string('split_file', '../cachedir/associative3d_test_set.txt', 'split file')
flags.DEFINE_string('rel_pose_file', '../cachedir/rel_poses_v3_val.pkl', 'predicted camera pose')
flags.DEFINE_integer('base', 0, 'run from line W[base]')
flags.DEFINE_integer('total', -1, 'run for [total] examples')
flags.DEFINE_integer('num_process', 8, 'number of processes')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus')
flags.DEFINE_boolean('gtbbox', False, 'use ground truth bounding box')
flags.DEFINE_boolean('save_results', True, 'save final results')
flags.DEFINE_boolean('save_objs', False, 'save visualizations')

# stitching hyperparameters
flags.DEFINE_integer('lambda_nomatch', 1, '')
flags.DEFINE_integer('lambda_rots', 5, '')
flags.DEFINE_integer('lambda_trans', 1, '')
flags.DEFINE_integer('lambda_af', 5, '')
flags.DEFINE_boolean('random_search', False, '')
#lambda_nomatch = 1
#lambda_rots = 5
#lambda_trans = 1
#lambda_af = 5

# command line args
opts = flags.FLAGS
opts.batch_size = 1
opts.name = 'exp22_10-4_wmse'
opts.num_train_epoch = 9
opts.classify_rot = True
opts.classify_dir = True
opts.pred_voxels = False
opts.use_context = True
opts.pred_labels=True
opts.upsample_mask=True
opts.pred_relative=True
opts.use_mask_in_common=True
opts.use_spatial_map=True
opts.pretrained_shape_decoder=True
opts.do_updates=True
opts.dwr_model=True
if opts.classify_rot:
    opts.nz_rot = 24
else:
    opts.nz_rot = 4



vis_list = [
    '01bc4f8cdc1fe117d4811ed0812c8145_000003_000013',
    '01fed431ceb06e4bc320e734b0964e2b_000002_000000',
    '02fa369fcb387eca76c0635507e0ed15_000009_000016',
    '0a41fac2fa30b5eeb934d1ae94340087_000009_000029',
    '0ac94d13bf5dc87a0ddc6ec84685ddfc_000018_000009',
    '0bb0df701c11f65eea205c2e24970817_000006_000000',
    '0d05b1c41404736ad97e7f7a4f4e7a0a_000007_000004',
    '0e506f624a05a5b44c9f5e3899385d6e_000004_000002',
    '1f254df66c240fee09b01d2fb15fa142_000004_000008',
    '1f76f38cdc1b0a58282016b4af7bc40b_000020_000007',
    '2ac722fd6ae4fdaab442679391414073_000011_000004',
    '2c4e0cb1530727da521a458ccc4485b6_000007_000008',
    '3c270eecc88f8fe0225ad1ae881b825e_000006_000003',
    # suncg additional wall
    '3b8a8710cb37fa2cb6eac3a04821a067_000003_000000',
    '3d0004835c700a1850001b772d8194fd_000000_000003',
    '3e78f0c28c8a51a9571d456b874409ca_000016_000028',
    '3e78f0c28c8a51a9571d456b874409ca_000016_000028',
    '3ef8758c5b3ff7860ee8313f43a6aa5c_000008_000021',
    '3f0c23f1a8c17d9e62a7c1f0c7a6fa31_000007_000011',
    '2d2a64c8c4f84b8ada8cd3e79b849818_000007_000011',
    '2e2840bdc0f8794f41aa425029895d6b_000001_000006',
    '2e5cf189c5348060c28f93305c02519e_000008_000006',
    '2ec0a760686672734ff10a0527c466c9_000006_000011',
    '2f8c5ff62c085c0ad7f3a1cc0302cb89_000045_000020',
]


def prepare_inputs(house_name, view_id, meta_loader=None):
    # read img
    img_path = os.path.join(opts.suncg_dir, 'renderings_ldr', house_name, view_id + '_mlt.png')
    img = imageio.imread(img_path)

    # preprocess images
    #img_fine = cv2.resize(img, (opts.img_width_fine, opts.img_height_fine))
    img_fine = np.array(Image.fromarray(img).resize((opts.img_width_fine, opts.img_height_fine), Image.BILINEAR))
    img_fine = np.transpose(img_fine, (2, 0, 1))
    #img_coarse = cv2.resize(img, (opts.img_width, opts.img_height))
    img_coarse = np.array(Image.fromarray(img).resize((opts.img_width, opts.img_height), Image.BILINEAR))
    img_coarse = np.transpose(img_coarse, (2, 0, 1))

    # read bbox
    gt_bbox_path = os.path.join(opts.suncg_dir, 'bboxes_node', house_name, 
                                view_id + '_bboxes.mat')
    gt_bbox = sio.loadmat(gt_bbox_path)
    if opts.gtbbox:
        temp = gt_bbox
    else:
        # edgebox proposals
        #proposal_path = os.path.join(opts.suncg_dir, 'edgebox_proposals',
        #                             house_name, view_id + '_proposals.mat')
        #temp = sio.loadmat(proposal_path)

        # faster-rcnn proposals
        proposal_path = os.path.join(opts.suncg_dir, 'faster_rcnn_proposals',
                                     house_name, view_id + '_proposals.mat')
        if os.path.exists(proposal_path):
            temp = sio.loadmat(proposal_path)
        else:
            temp = {'node_ids': np.array([]), 'bboxes': np.array([]), 'scores': np.array([])}
        
    if opts.gtbbox:
        assert(meta_loader is not None)
        house_data = suncg_parse.load_json(
            os.path.join(opts.suncg_dir, 'house', house_name, 'house.json'))
        # _, objects_bboxes = suncg_parse.select_ids(
        #     house_data, gt_bbox, meta_loader=meta_loader, min_pixels=500)
        house_copy, objects_bboxes, node_ids, obj_nodes = suncg_parse.select_ids_multi(
            house_data, gt_bbox, meta_loader=meta_loader, min_pixels=500)

        objects_bboxes -= 1
        proposals = objects_bboxes.astype(np.float32)
        scores = np.ones(proposals.shape[0], dtype=np.float32)
    else:
        # edgebox
        #proposals = temp['proposals'][:, 0:4]

        # faster-rcnn proposals
        proposals = temp['bboxes'].astype(np.int)
        #proposals[proposals < 0] = 0
        scores = temp['scores']
        # node_ids are invalid in detection
        node_ids = [0 for _ in range(len(scores))]
        
    # prepare inputs
    inputs = {}
    inputs['img'] = torch.from_numpy(img_coarse/255.0).unsqueeze(0)
    inputs['img_fine'] = torch.from_numpy(img_fine/255.0).unsqueeze(0)
    inputs['bboxes'] = [torch.from_numpy(proposals)]
    inputs['scores'] = [torch.from_numpy(scores)]
    inputs['empty'] = False
    inputs['node_ids'] = np.array(node_ids)
    if opts.gtbbox:
        inputs['obj_nodes'] = obj_nodes
    
    return inputs, img


def uncollate_pred(code_vars):
    n_rois = code_vars['shape'].size()[0]
    code_list = suncg_parse.uncollate_codes(code_vars, 1, torch.Tensor(n_rois).fill_(0))
    return code_list


def forward_single_view(tester, house_name, view_id, meta_loader=None, mesh_dir=None, prefix='view1'):
    # prepare inputs
    inputs, img = prepare_inputs(house_name, view_id, meta_loader=meta_loader)
    tester.set_input(inputs)

    # inference
    objects, node_ids = tester.predict_box3d()

    # visualization
    if mesh_dir is not None:
        if tester.rois_pos_vis is not None:
            bbox1 = tester.rois_pos_vis.cpu().numpy()
            roi_vis = render_utils.vis_detections(img, bbox1[:, 1:])
        else:
            roi_vis = img
        imageio.imwrite(os.path.join(mesh_dir, '{}_{}_rois.png'.format(prefix, view_id)), roi_vis)
        #tester.save_codes_mesh(mesh_dir, objects, 'A_code_pred')

    # uncollate prediction
    objects = uncollate_pred(objects)[0]
    
    # set node_id in codes
    if opts.gtbbox:
        for i in range(len(objects)):
            objects[i]['node_ids'] = [node_ids[i]]
        for i in range(len(objects)):
            objects[i]['obj_node'] = inputs['obj_nodes'][i]
    else:
        for i in range(len(objects)):
            objects[i]['scores'] = [inputs['scores'][0][0][i].item()]
    
    return objects, node_ids


def get_gt_codes(obj_loader, meta_loader, house_name, view_ids):
    # get house data
    house_data = suncg_parse.load_json(
        os.path.join(opts.suncg_dir, 'house', house_name, 'house.json'))
        
    # merge bboxes from different views
    # Concatenating bbox actually makes no sense. Just want to make 
    # suncg_parse.select_ids work.
    node_ids = []
    bboxes = []
    n_pixels = []
    for view_id in view_ids:
        cam_file = os.path.join(opts.suncg_dir, 'camera', house_name, 'room_camera.txt')
        camera_poses = suncg_parse.read_camera_pose(cam_file)
        campose = camera_poses[int(view_id)]
        cam2world = suncg_parse.campose_to_extrinsic(campose).astype(np.float32)
        world2cam = scipy.linalg.inv(cam2world).astype(np.float32)
        bbox_data_view = scipy.io.loadmat(
            os.path.join(opts.suncg_dir, 'bboxes_node', house_name, 
                         view_id + '_bboxes.mat')
        )
        for idx, node_id in enumerate(bbox_data_view['node_ids'][:, 0]):
            if node_id in node_ids:
                continue
            node_ids.append(node_id)
            bboxes.append(bbox_data_view['bboxes'][idx])
            n_pixels.append(bbox_data_view['n_pixels'][idx])
    bbox_data = {
        'node_ids': np.array(node_ids)[:, np.newaxis],
        'bboxes': np.array(bboxes),
        'n_pixels': np.array(n_pixels),
    }

    # get objects
    #objects_data, objects_bboxes = suncg_parse.select_ids(
    #    house_data, bbox_data, meta_loader=meta_loader, min_pixels=500)
    objects_data, objects_bboxes, select_node_ids, _ = suncg_parse.select_ids_multi(
        house_data, bbox_data, meta_loader=meta_loader, min_pixels=500)
        
    objects_codes, _ = suncg_parse.codify_room_data(
        objects_data, world2cam, obj_loader,
        max_object_classes=max_object_classes)

    objects_bboxes -= 1 #0 indexing to 1 indexing
    if len(objects_codes) > max_rois:
        select_inds = np.random.permutation(len(objects_codes))[0:max_rois]
        objects_bboxes = objects_bboxes[select_inds, :].copy()
        objects_codes = [objects_codes[ix] for ix in select_inds]

    for i in range(len(objects_codes)):
        objects_codes[i]['node_ids'] = [select_node_ids[i]]
            
    return objects_codes


def forward_affinity_gt(select_node_ids1, select_node_ids2):
    """
    Calculate affinity matrix according to select_node_ids
    """
    affinity = np.zeros((max_rois, max_rois), dtype=np.float32)
    for i, node_id1 in enumerate(select_node_ids1):
        for j, node_id2 in enumerate(select_node_ids2):
            if node_id1 == node_id2:
                affinity[i, j] = 1
    return affinity


def getErrTran(gt, pred):
    return np.linalg.norm(pred-gt)


def getErrRot(gt, pred):
    d = np.array(np.abs(np.sum(np.multiply(pred, gt))))
    if d > 1:
        d = 1
    if d < -1:
        d = -1
    return np.arccos(d) * 180/math.pi


def predict(mesh_dir, house_name, view1, view2, tester, rel_pose, 
            stitching='naive', gt_pose=False, gt_codes=False, 
            obj_loader=None, meta_loader=None, save_prefix='codes',
            save_a_prefix=None, save_b_prefix=None, gt_affinity=False, 
            save_rel_pose=False):
    if gt_pose and gt_codes:
        assert(obj_loader is not None)
        assert(meta_loader is not None)
        codes = get_gt_codes(obj_loader, meta_loader, house_name, [view1, view2])
        if opts.save_objs:
            save_codes_mesh(mesh_dir, codes, prefix=save_prefix)
        return

    # load codes
    if gt_codes:
        assert(obj_loader is not None)
        assert(meta_loader is not None)
        objects1 = get_gt_codes(obj_loader, meta_loader, house_name, [view1])
        objects2 = get_gt_codes(obj_loader, meta_loader, house_name, [view2])
    else:
        try:
            objects1, select_node_ids_1 = forward_single_view(tester, house_name, view1,
                meta_loader=meta_loader, mesh_dir=mesh_dir, prefix='1')
            objects2, select_node_ids_2 = forward_single_view(tester, house_name, view2,
                meta_loader=meta_loader, mesh_dir=mesh_dir, prefix='2')
        except RuntimeError as e:
            print("ERR {} {} {} {}".format(e, house_name, view1, view2))
            return

        gt_affinity_matrix = forward_affinity_gt(select_node_ids_1, select_node_ids_2)
        if gt_affinity_matrix is None:
            print("ERR gt_affinity_matrix is None at {} {} {}".format(house_name, view1, view2))
            import pdb;pdb.set_trace()
        max_sz = max(len(select_node_ids_1), len(select_node_ids_2))
        if max_sz < 5:
            max_sz = 5
        elif max_sz < 10:
            max_sz = 10
        affinity_vis = gt_affinity_matrix[:max_sz, :max_sz]
        plt.figure()
        sns.heatmap(affinity_vis, vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_gt.png'))
        plt.close()
        
    # insert colormap
    for obj in objects1:
        obj['cmap'] = 'red'
    for obj in objects2:
        obj['cmap'] = 'blue'
    
        
    if save_a_prefix is not None and opts.save_objs:
        save_codes_mesh(mesh_dir, objects1, prefix=save_a_prefix)
        #for id, obj in enumerate(objects1):
        #   save_codes_mesh(mesh_dir, [obj], prefix=save_a_prefix + '_' + str(id))
    if save_b_prefix is not None and opts.save_objs:
        save_codes_mesh(mesh_dir, objects2, prefix=save_b_prefix)
        #for id, obj in enumerate(objects2):
        #   save_codes_mesh(mesh_dir, [obj], prefix=save_b_prefix + '_' + str(id))

    # load relpose
    if gt_pose:
        rel_rot = rel_pose['rot_gt']
        rel_tran = rel_pose['tran_gt']
    else:
        rel_rot = rel_pose['rot_pred']
        rel_tran = rel_pose['tran_pred']
        rot_logits = rel_pose['rot_logits']
        tran_logits = rel_pose['tran_logits']
    
    # stitch policy
    try:
        if stitching == 'naive':
            # merge = False
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_naive(objects1, objects2, rel_rot, rel_tran)
        elif stitching == 'singleview_left':
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_singleview_left(objects1, objects2, rel_rot, rel_tran)
        elif stitching == 'singleview_right':
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_singleview_right(objects1, objects2, rel_rot, rel_tran)
        elif stitching == 'singleview_random':
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_singleview_random(objects1, objects2, rel_rot, rel_tran)
        elif stitching == 'semantic':
            # category level stitching
            assert opts.gtbbox is True
            assert(obj_loader is not None)
            assert(meta_loader is not None)
            objects1_gt = get_gt_codes(obj_loader, meta_loader, house_name, [view1])
            objects2_gt = get_gt_codes(obj_loader, meta_loader, house_name, [view2])
            stitch_objects_semantic(objects1, objects2, objects1_gt, objects2_gt,
                rel_rot, rel_tran, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix)
            return
        elif stitching == 'nms':
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_nms(objects1, objects2, rel_rot, rel_tran)
        elif stitching == 'affinity':
            # merge = True, affinity = True
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_affinity(objects1, objects2, rel_rot, 
                rel_tran, mesh_dir=mesh_dir)
        elif stitching == 'af_chamfer':
            # merge = True, affinity = true, chamfer = true, search_RT = True
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_af_chamfer(objects1, objects2, rot_logits, 
                tran_logits, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity = gt_affinity)
        elif stitching == 'af_matching':
            # merge = True, affinity = true, chamfer = True
            codes = stitch_objects_af_matching(objects1, objects2, rel_rot, 
                rel_tran, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix)
        elif stitching == 'rand_chamfer':
            # randomized optimization
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_rand_chamfer(objects1, objects2, rot_logits, 
                tran_logits, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity = gt_affinity)
        elif stitching == 'rand_chamfer_prob':
            # randomized optimization considering the probability
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_rand_chamfer_prob(objects1, objects2, rot_logits, 
                tran_logits, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity = gt_affinity)
        elif stitching == 'chamfer_edge':
            # randomized optimization considering the probability
            #lambds = [lambda_nomatch, lambda_rots, lambda_trans, lambda_af]
            
            if gt_pose:
                codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_edge_wpose(objects1, objects2, rel_rot, 
                    rel_tran, tester, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity=gt_affinity, opts=opts)
            else:
                codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_edge(objects1, objects2, rot_logits, 
                    tran_logits, tester, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity=gt_affinity, opts=opts)

        elif stitching == 'rand_chamfer_prob_shape':
            # rand_chamfer_prob + average shape code
            codes, rel_tran_after_stitch, rel_rot_after_stitch, affinity_info = stitch_objects_asso_shape(objects1, objects2, rot_logits, 
                tran_logits, tester, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity=gt_affinity)
        elif stitching == 'rand_chamfer_ambiguity':
            # randomized optimization considering the probability
            codes_options = stitch_objects_rand_chamfer_ambiguity(objects1, objects2, rot_logits, 
                tran_logits, mesh_dir=mesh_dir, gt_affinity_m=gt_affinity_matrix, use_gt_affinity = gt_affinity)
            if opts.save_objs:
                for i, codes in enumerate(codes_options):
                    save_codes_mesh(mesh_dir, codes, prefix=save_prefix + '_{}'.format(i))
            return
        elif stitching == 'generate_vt':
            codes = generate_volume_trans(objects1, objects2, rel_rot, 
                rel_tran, mesh_dir=mesh_dir)
        else:
            raise NotImplementedError
    except RuntimeError as e:
        print("ERR {} {} {} {}".format(e, house_name, view1, view2))
        return

    if opts.save_results:
        affinity_pred = affinity_info['affinity_pred']
        affinity_gt = affinity_info['affinity_gt']
        matching_pred = affinity_info['matching']
        assert(obj_loader is not None)
        assert(meta_loader is not None)
        codes_gt = get_gt_codes(obj_loader, meta_loader, house_name, [view1, view2])
        #codes_gt = get_gt_codes(obj_loader, meta_loader, house_name, [view1])
        save_results(mesh_dir, codes, codes_gt, rel_pose, 
                     rel_tran_after_stitch, rel_rot_after_stitch, 
                     affinity_pred, affinity_gt, matching_pred)

    if opts.save_objs:
        save_codes_mesh(mesh_dir, codes, prefix=save_prefix)

    if save_rel_pose:
        with open(os.path.join(mesh_dir, 'rel_pose.txt'), 'w') as f:
            pred_rot = rel_pose['rot_pred']
            pred_tran = rel_pose['tran_pred']
            gt_rot = rel_pose['rot_gt']
            gt_tran = rel_pose['tran_gt']
            gt_tran_in_search, gt_rot_in_search = is_gt_tran_rot_in_search_space(
                gt_tran=gt_tran, gt_rot=gt_rot, tran_logits=rel_pose['tran_logits'], rot_logits=rel_pose['rot_logits'])

            
            f.write('error_in_top1_tran: {}\n'.format(getErrTran(gt_tran, pred_tran)))
            f.write('error_in_top1_rot: {}\n'.format(getErrRot(gt_rot, pred_rot)))
            f.write('pred_top1_tran: {}\n'.format(rel_pose['tran_pred']))
            f.write('pred_top1_rot: {}\n'.format(rel_pose['rot_pred']))
            f.write(stitching + '_tran: {}\n'.format(getErrTran(gt_tran, rel_tran_after_stitch)))
            f.write(stitching + '_rot: {}\n'.format(getErrRot(gt_rot, rel_rot_after_stitch)))
            f.write(stitching + '_tran: {}\n'.format(rel_tran_after_stitch))
            f.write(stitching + '_rot: {}\n'.format(rel_rot_after_stitch))
            f.write('gt_tran_in_search: {}\n'.format(gt_tran_in_search))
            f.write('gt_rot_in_search: {}\n'.format(gt_rot_in_search))
            f.write('gt_tran_err: {}\n'.format(getErrTran(gt_tran, class2tran(xyz2class(*gt_tran)))))
            f.write('gt_rot_err: {}\n'.format(getErrRot(gt_rot, class2quat(quat2class(*gt_rot)))))
            f.write('psedo_gt_tran_class: {}\n'.format(xyz2class(*gt_tran)))
            f.write('psedo_gt_rot_class: {}\n'.format(quat2class(*gt_rot)))
            f.write('gt_match_in_proposal: {}\n'.format(affinity_info['gt_match_in_proposal']))
            f.write('hit_gt_match: {}\n'.format(affinity_info['hit_gt_match']))
            
    
def run(lines, output_dir, queue=None, gpu_id=0):
    # set gpu id
    torch.cuda.set_device(gpu_id)

    # set random seed
    if queue is not None:
        random.seed(2019)
        np.random.seed(2019)
        torch.manual_seed(2019)

    # load args
    args = opts.read_flags_from_files(('--flagfile=' + os.path.join(output_dir, 'args.cfg'), ))
    opts._parse_args(args, True)

    # load pre-trained model
    ckpt_file = '../cachedir/snapshots/{}/pred_net_{}.pth'.format(opts.name, opts.num_train_epoch)
    #clean_checkpoint_file(ckpt_file)
    tester = demo_utils.DemoTester(opts)
    tester.init_testing()

    # load relative camera pose
    with open(opts.rel_pose_file, 'rb') as f:
        rel_poses = pickle.load(f)
    
    # initialize meta_loader and obj_loader
    meta_loader = suncg_parse.MetaLoader(
        os.path.join(opts.suncg_dir, 'ModelCategoryMappingEdited.csv')
    )
    obj_loader = suncg_parse.ObjectLoader(os.path.join(opts.suncg_dir, 'object'))
    obj_loader.preload()
    
    for line in tqdm(lines, total=len(lines), disable=(queue is not None)):
        # parse line
        splits = line.split(' ')
        img1_path = splits[0]
        img2_path = splits[8]
        house_name = img1_path.split('/')[-2]
        img1_name = img1_path.split('/')[-1]
        img2_name = img2_path.split('/')[-1]
        view1 = img1_name[:6]
        view2 = img2_name[:6]

        pair_id = house_name + '_' + view1 + '_' + view2
        #if pair_id not in vis_list:
        #    continue

        # set up mesh_dir
        mesh_dir = os.path.join(opts.output_dir, house_name + '_' + view1 + '_' + view2)
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        else:
            pass
            #continue

        # set up relpose
        rel_pose = rel_poses[pair_id]

        # Ground Truth Voxels
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
            stitching='naive', gt_pose=True, gt_codes=True,
            obj_loader=obj_loader, meta_loader=meta_loader,
            save_prefix='codes_gt_pose_gt')

        # single view left
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='singleview_left', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """
        
        # single view right
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='singleview_right', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # single view random
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='singleview_random', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Baseline 1
        # top1 rotation, top1 translation, no stitching
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='naive', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='naive',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Baseline 2
        # top1 rotation, top1 translation, NMS
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='nms', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='nms',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Baseline 3
        # top1 rotation, top1 translation, top1 affinity
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='affinity', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='affinity',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Our CVPR method
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='rand_chamfer_prob', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Our ECCV method (better and faster)
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='chamfer_edge', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='ours',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # our method with gt
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='chamfer_edge', gt_pose=True, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='ours',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)

        # our method + average shape code
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='rand_chamfer_prob_shape', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # Ambiguity
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='rand_chamfer_ambiguity', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # cagetory level matching
        """
        predict(mesh_dir, house_name, view1, view2, tester, rel_pose,
                stitching='semantic', gt_pose=False, gt_codes=False,
                obj_loader=obj_loader, meta_loader=meta_loader, gt_affinity=False,
                save_prefix='codes_pred_pose_pred_affinity_pred',
                save_a_prefix='view1', save_b_prefix='view2', save_rel_pose=True)
        """

        # done
        if queue is None:
            tqdm.write(pair_id)
        else:
            queue.put(pair_id)


def main(_):
    #global lambda_nomatch, lambda_rots, lambda_trans, lambda_af
    if opts.random_search:
        opts.lambda_nomatch = random.randint(1, 5)
        opts.lambda_rots = 5
        opts.lambda_trans = 1
        opts.lambda_af = random.randint(1, 30)


    #print(opts)
    print("Results saved in: {}".format(opts.output_dir))
    with open(os.path.join(opts.output_dir, 'para.txt'), 'w') as f:
        f.write('lambda_nomatch {}\n'.format(opts.lambda_nomatch))
        f.write('lambda_rots {}\n'.format(opts.lambda_rots))
        f.write('lambda_trans {}\n'.format(opts.lambda_trans))
        f.write('lambda_af {}\n'.format(opts.lambda_af))
    opts.append_flags_into_file(os.path.join(opts.output_dir, 'args.cfg'))
    f = open(opts.split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    #random.shuffle(lines)
    base = opts.base
    total = opts.total
    if total == -1:
        total = len(lines) - base
    lines = lines[base:(base + total)]

    # single process
    if opts.num_process == 0:
        run(lines, opts.output_dir)
        return
    
    # multiprocess
    multiprocessing.set_start_method('spawn')
    n = math.ceil(len(lines) / opts.num_process)
    lines = [lines[i:i + n] for i in range(0, len(lines), n)]
    jobs = []
    queue = Queue()
    assert opts.num_gpus <= torch.cuda.device_count()
    num_p_per_gpu = math.ceil(opts.num_process / opts.num_gpus)
    for i in range(opts.num_process):
        gpu_id = i // num_p_per_gpu
        if i >= len(lines):
            continue
        p = Process(target=run, args=(lines[i], opts.output_dir, queue, gpu_id))
        p.start()
        jobs.append(p)

    for i in tqdm(range(total)):
        tqdm.write(queue.get())
    
    for job in jobs:
        job.join()


if __name__=='__main__':
    app.run(main)
