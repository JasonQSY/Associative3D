"""
Evaluation using detection.
"""
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import pdb
import os
import sys
import torch
from copy import deepcopy
import open3d
import mcubes
import pickle

code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_root)
from relative3d.utils import bbox_utils, metrics
from relative3d.renderer import utils as render_utils
import evaluate_detection


parser = argparse.ArgumentParser()
parser.add_argument('--split_file', type=str, 
    default='../script/v3_rpnet_split_relative3d/v3_validation_subset_MGT0.txt', 
    help='split file path')
parser.add_argument('--result_dir', type=str, default='/x/syqian/mesh_output', help='results directory')
args = parser.parse_args()


EP_rot_delta_thresh =   [30.   , 30.      , 400.   , 30.      , 30.      , 30.         , 400.          , 400.          , 400.          , 30                , ]
EP_trans_delta_thresh = [1.    , 1.       , 1.     , 1000.    , 1.       , 1000.       , 1.            , 1000.         , 1000.         , 1000.             , ]
EP_shape_iou_thresh =   [0.25  , 0        , 0.25   , 0.25     , 0.25     , 0           , 0             , 0.25          , 0             , 0.25              , ]
EP_scale_delta_thresh = [0.2   , 0.2      , 0.2    , 0.2      , 100.     , 100         , 100           , 100           , 0.2           , 100               , ]
EP_ap_str =             ['all' , '-shape' , '-rot' , '-trans' , '-scale' , 'rot'       , 'trans'       , 'shape'       , 'scale'       , 'rot+shape'       , ]


iou_thres = 0.25
trans_thres = 1.25
rot_thres = 30
scale_thres = 0.2
pcd_thres = 0.05


EP_shape_iou_thresh = np.array(EP_shape_iou_thresh)
EP_shape_iou_thresh[EP_shape_iou_thresh > 0.01] = iou_thres
EP_shape_iou_thresh = EP_shape_iou_thresh.tolist()
EP_trans_delta_thresh = np.array(EP_trans_delta_thresh)
EP_trans_delta_thresh[EP_trans_delta_thresh < 380] = trans_thres
EP_trans_delta_thresh = EP_trans_delta_thresh.tolist()
EP_rot_delta_thresh = np.array(EP_rot_delta_thresh)
EP_rot_delta_thresh[EP_rot_delta_thresh < 380] = rot_thres
EP_rot_delta_thresh = EP_rot_delta_thresh.tolist()
EP_scale_delta_thresh = np.array(EP_scale_delta_thresh)
EP_scale_delta_thresh[EP_scale_delta_thresh < 90] = scale_thres
EP_scale_delta_thresh = EP_scale_delta_thresh.tolist()


def volume_iou(pred, gt, thresh=0.5):
    pred = render_utils.downsample(pred, pred.shape[0] // 32)
    gt = render_utils.downsample(gt, gt.shape[0] // 32)
    pred = torch.FloatTensor(pred)
    gt = torch.FloatTensor(gt)
    gt = gt.float().ge(0.5)
    pred = pred.float().ge(thresh)
    intersection = torch.mul(gt, pred).sum()
    union = gt.sum() + pred.sum() - intersection
    intersection = intersection.item()
    union = union.item()
    return intersection/union


def voxel_grid_to_mesh(vox_grid):
    '''Converts a voxel grid represented as a numpy array into a mesh.'''
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
       sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1,1),(1,1),(1,1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    return out_mesh


def voxel_grid_to_pcd(vox_grid):
    '''Converts a voxel grid represented as a numpy array into a mesh.'''
    vox_grid = render_utils.downsample(vox_grid, vox_grid.shape[0] // 32)
    vox_grid = (vox_grid > 0.5).astype(np.float32)
    if vox_grid.sum() <= 3: 
        return None

    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
       sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1,1),(1,1),(1,1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    #pcd = out_mesh.sample_points_poisson_disk(100)
    #pdb.set_trace()
    pcd = out_mesh.sample_points_uniformly(1000)
    #points = np.asarray(pcd.points)
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    #d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    #d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    #pdb.set_trace()
    return fscore, precision, recall


def volume_fscore(pred, gt, thresh=0.5):
    if pred is None or gt is None:
        return 0.
    #pred_pcd = voxel_grid_to_pcd(pred)
    #gt_pcd = voxel_grid_to_pcd(gt)
    fscore, _, _ = calculate_fscore(gt, pred, th=pcd_thres)
    return fscore


def evaluate(results):
    # Get Predictions.
    objects_pred = results['objects_pred']
    matching_pred = results['matching_pred']

    #affinity_gt = results['affinity_gt']
    #print(affinity_gt[:6, :6])

    if matching_pred is not None:
        matching_pred = np.array(matching_pred)
        num_match = (matching_pred > 0).sum()
        affinity_pred = results['affinity_pred']
    objects_pred.sort(key=lambda obj: -np.array(obj['scores']).max())
    shapes = []
    scales = []
    rots = []
    trans = []
    scores = []
    #boxes = []
    for i, obj in enumerate(objects_pred):
        shape = obj['shape']
        shape = voxel_grid_to_pcd(shape)
        shapes.append(shape)
        scales.append(obj['scale'])
        rots.append(obj['quat'])
        trans.append(obj['trans'])
        score = np.array(obj['scores']).max()

        """
        if matching_pred is not None:  # adjust score
            if i < num_match:
                idx = np.where(matching_pred > 0)[0][i]
                #score = 1 - (1 - affinity_pred[idx, matching_pred[idx]]) * score
                score = score / 2 + affinity_pred[idx, matching_pred[idx]]
            else:
                score = score
        """

        scores.append(score)
        #scores.append(1.0)
    #shapes = np.array(shapes)
    scales = np.array(scales)
    rots = np.array(rots)
    trans = np.array(trans)
    scores = np.array(scores)
    scores = scores[:, np.newaxis]

    # Get Ground Truth.
    objects_gt = results['objects_gt']
    gt_shapes = []
    gt_scales = []
    gt_rots = []
    gt_trans = []
    #boxes = []
    for obj in objects_gt:
        shape = obj['shape']
        shape = voxel_grid_to_pcd(shape)
        gt_shapes.append(shape)
        gt_scales.append(obj['scale'])
        gt_rots.append(obj['quat'])
        gt_trans.append(obj['trans'])
    #gt_shapes = np.array(gt_shapes)
    gt_scales = np.array(gt_scales)
    gt_rots = np.array(gt_rots)
    gt_trans = np.array(gt_trans)

        
    if len(objects_pred) == 0:
        stats = []
        for i in range(len(EP_ap_str)):
            tp = np.zeros((0, 1), dtype=bool)
            fp = np.zeros((0, 1), dtype=bool)
            sc = np.zeros((0, 1), dtype=bool)
            num_inst = len(objects_gt)
            stats.append([tp, fp, sc, num_inst, None, None, None])
        #tqdm.write(str(0.0))
        return stats


    #iou_box = bbox_utils.bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

    err_trans = np.linalg.norm(np.expand_dims(trans, 1) - np.expand_dims(gt_trans, 0), axis=2)
    err_scales = np.mean(np.abs(np.expand_dims(np.log(scales), 1) - np.expand_dims(np.log(gt_scales), 0)), axis=2)
    err_scales /= np.log(2.0)

    ndt, ngt = err_scales.shape
    err_shapes = err_scales * 0.
    err_rots = err_scales * 0.
    #iou_box = np.ones_like(err_scales)# / 2
    #for i in range(ngt):
    #    iou_box[i][i] = 1.0

    for i in range(ndt):
        for j in range(ngt):
            #err_shapes[i,j] = volume_iou(shapes[i], gt_shapes[j])
            err_shapes[i,j] = volume_fscore(shapes[i], gt_shapes[j])
            q_errs = []
            for gt_quat in gt_rots[j]:
                q_errs.append(metrics.quat_dist(rots[i], gt_quat))
            err_rots[i,j] = min(q_errs)

    #print(err_shapes)
    """
    
    for i in range(ndt):
        out = shapes[i] > 0.5
        out = voxel_grid_to_mesh(out)
        open3d.io.write_triangle_mesh('output/det_{}.obj'.format(i).format(i), out)
        #vox = binvox_rw.Voxels(out, [32, 32, 32], [0,0,0], 1, 'xyz')
        #with open('output/det_{}.binvox'.format(i), 'wb') as f:
        #    vox.write(f)
    
    for i in range(ngt):
        out = gt_shapes[i]
        out = render_utils.downsample(out, out.shape[0] // 32)
        out = out > 0.5
        out = voxel_grid_to_mesh(out)
        open3d.io.write_triangle_mesh('output/gt_{}.obj'.format(i), out)
        
        #vox = binvox_rw.Voxels(out, [32, 32, 32], [0,0,0], 1, 'xyz')
        #with open('output/gt_{}.binvox'.format(i), 'wb') as f:
        #    vox.write(f)
    """

    # Run the benchmarking code here.
    #threshs = [EP_box_iou_thresh, EP_rot_delta_thresh, EP_trans_delta_thresh, EP_shape_iou_thresh, EP_scale_delta_thresh]
    #fn = [np.greater_equal, np.less_equal, np.less_equal, np.greater_equal, np.less_equal]
    #overlaps = [iou_box, err_rots, err_trans, err_shapes, err_scales]
    threshs = [EP_rot_delta_thresh, EP_trans_delta_thresh, EP_shape_iou_thresh, EP_scale_delta_thresh]
    fn = [np.less_equal, np.less_equal, np.greater_equal, np.less_equal]
    overlaps = [err_rots, err_trans, err_shapes, err_scales]

    _dt = {'sc': scores}
    _gt = {'diff': np.zeros((ngt,1), dtype=np.bool)}
    _bopts = {'minoverlap': 0.5}
    stats = []
    for i in range(len(EP_ap_str)):
        # Compute a single overlap that ands all the thresholds.
        ov = []
        for j in range(len(overlaps)):
            ov.append(fn[j](overlaps[j], threshs[j][i]))
        _ov = np.all(np.array(ov), 0).astype(np.float32)
        # Benchmark for this setting.
        tp, fp, sc, num_inst, dup_det, inst_id, ov = evaluate_detection.inst_bench_image(_dt, _gt, _bopts, _ov)
        #if EP_ap_str[i] == 'all':
        #pdb.set_trace()
        #tqdm.write(str(tp.sum() / tp.shape[0]))
        stats.append([tp, fp, sc, num_inst, dup_det, inst_id, ov])
    return stats



def main():
    f = open(args.split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    #lines = lines[:5]
    eval_path = os.path.join(args.result_dir, 'eval.txt')

    """
    if os.path.exists(eval_path):
        print("[detect] eval results exists.")
        f = open(eval_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            print(line)
        return
    """

    bench_stats = []
    for line in tqdm(lines):
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
        #tqdm.write(pair_id)

        # read results
        try:
            result_dir = os.path.join(args.result_dir, pair_id)
            rf = open(os.path.join(result_dir, 'results.pkl'), 'rb')
            results = pickle.load(rf)
            rf.close()
        except FileNotFoundError as e:
            raise e

        bench_image_stats = evaluate(results)
        bench_stats.append(bench_image_stats)

    # Accumulate stats, write to FLAGS.results_eval_dir.
    bb = list(zip(*bench_stats))
    #pdb.set_trace()
    bench_summarys = []
    eval_params = {}
    #f = open(eval_path, 'w')
    for i in range(len(EP_ap_str)):
        tp, fp, sc, num_inst, dup_det, inst_id, ov = zip(*bb[i])
        ap, rec, prec, npos, details = evaluate_detection.inst_bench(None, None, None, tp, fp, sc, num_inst)
        bench_summary = {'prec': prec.tolist(), 'rec': rec.tolist(), 'ap': ap[0], 'npos': npos}
        print('{:>20s}: {:5.3f}'.format(EP_ap_str[i], ap[0]*100.))
        #f.write('{:>20s}: {:5.3f}\n'.format(EP_ap_str[i], ap[0]*100.))
        bench_summarys.append(bench_summary)

    #f.close()


if __name__=='__main__':
    main()
