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
import cv2
import random
import shutil
from pyquaternion import Quaternion
import pickle
import itertools
from copy import deepcopy
from PIL import Image
import math
from time import time
from absl import app, flags

code_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(code_root, '..'))
from object_branch.renderer import utils as render_utils
from object_branch.utils import suncg_parse
from object_branch.stitching.stitch_utils import *
from object_branch.utils import metrics
opts = flags.FLAGS


def stitch_objects_singleview_left(objects1, objects2, rel_rot, rel_tran):
    """
    only keep view 1
    """
    rel_quat = Quaternion(rel_rot)
    for obj in objects1:
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        
    codes = objects1

    affinity_info = {}
    affinity_info['hit_gt_match'] = None
    affinity_info['gt_match_in_proposal'] = None
    affinity_info['affinity_pred'] = None
    affinity_info['affinity_gt'] = None
    affinity_info['matching'] = None

    return codes, rel_tran, rel_rot, affinity_info


def stitch_objects_singleview_random(objects1, objects2, rel_rot, rel_tran):
    """
    keep view 1 or view 2 randomly
    """
    if random.random() > 0.5:
        return stitch_objects_singleview_right(objects1, objects2, rel_rot, rel_tran)
    else:
        return stitch_objects_singleview_left(objects1, objects2, rel_rot, rel_tran)


def stitch_objects_singleview_right(objects1, objects2, rel_rot, rel_tran):
    """
    only keep view2
    """
    codes = objects2

    affinity_info = {}
    affinity_info['hit_gt_match'] = None
    affinity_info['gt_match_in_proposal'] = None
    affinity_info['affinity_pred'] = None
    affinity_info['affinity_gt'] = None
    affinity_info['matching'] = None

    return codes, rel_tran, rel_rot, affinity_info


def stitch_objects_naive(objects1, objects2, rel_rot, rel_tran):
    rel_quat = Quaternion(rel_rot)
    for obj in objects1:
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        
    codes = objects1 + objects2

    affinity_info = {}
    affinity_info['hit_gt_match'] = None
    affinity_info['gt_match_in_proposal'] = None
    affinity_info['affinity_pred'] = None
    affinity_info['affinity_gt'] = None
    affinity_info['matching'] = None

    return codes, rel_tran, rel_rot, affinity_info


def stitch_objects_semantic(objects1, objects2, objects1_gt, objects2_gt, rel_rot, rel_tran, mesh_dir=None, gt_affinity_m=None):
    """
    Category level stitching
    """
    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    affinity_pred = forward_affinity(id1, id2, mesh_dir=mesh_dir)

    results = {
        'mesh_dir': mesh_dir,
        'objects1': objects1,
        'objects2': objects2,
        'objects1_gt': objects1_gt,
        'objects2_gt': objects2_gt,
        'relpose': {
            'rotation': rel_rot,
            'translation': rel_tran,
        },
        'affinity_pred': affinity_pred,
        'affinity_gt': gt_affinity_m,
    }
    prefix = 'semantic'
    f = open(os.path.join(mesh_dir, prefix + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()


def stitch_objects_nms(objects1, objects2, rel_rot, rel_tran):
    rel_quat = Quaternion(rel_rot)
    for obj in objects1:
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran

    affinity_info = {}
    affinity_info['hit_gt_match'] = None
    affinity_info['gt_match_in_proposal'] = None
    affinity_info['affinity_pred'] = None
    affinity_info['affinity_gt'] = None
    affinity_info['matching'] = None
        
    #codes = objects1 + objects2
    if len(objects1) == 0 or len(objects2) == 0:
        codes = objects1 + objects2
        return codes, rel_tran, rel_rot, affinity_info


    def collect_info(objects):
        shapes = []
        scales = []
        rots = []
        trans = []
        for obj in objects:
            shapes.append(obj['shape'])
            scales.append(obj['scale'])
            rots.append(obj['quat'])
            trans.append(obj['trans'])
        shapes = np.array(shapes)
        scales = np.array(scales)
        rots = np.array(rots)#[:, 0]
        trans = np.array(trans)
        return shapes, scales, rots, trans

    shapes1, scales1, rots1, trans1 = collect_info(objects1)
    shapes2, scales2, rots2, trans2 = collect_info(objects2)

    err_trans = np.linalg.norm(np.expand_dims(trans1, 1) - np.expand_dims(trans2, 0), axis=2)
    err_scales = np.mean(np.abs(np.expand_dims(np.log(scales1), 1) - np.expand_dims(np.log(scales2), 0)), axis=2)
    err_scales /= np.log(2.0)
    ndt, ngt = err_scales.shape
    err_shapes = err_scales * 0.
    err_rots = err_scales * 0.
    iou_box = np.ones_like(err_scales)# / 2
    for i in range(ndt):
        for j in range(ngt):
            err_shapes[i,j] = volume_iou(shapes1[i], shapes2[j])
            #q_errs = []
            #for gt_quat in gt_rots[j]:
            #    q_errs.append(metrics.quat_dist(rots[i], gt_quat))
            #err_rots[i,j] = min(q_errs)
            err_rots[i, j] = metrics.quat_dist(rots1[i], rots2[j])

    ov = []
    #ov.append(err_trans < 1.)
    #ov.append(err_scales < 0.2)
    #ov.append(err_shapes > 0.25)
    #ov.append(err_rots < 30)
    #ov.append(err_trans < 1.)
    #ov.append(err_scales < 0.2)
    #ov.append(err_shapes > 0.5)
    #ov.append(err_rots < 10)

    ov.append(err_trans < 1.)
    ov.append(err_scales < 0.2)
    ov.append(err_shapes > 0.25)
    ov.append(err_rots < 30)

    _ov = np.all(np.array(ov), 0)#.astype(np.float32)
    merge_list = []
    for i in range(ndt):
        for j in range(ngt):
            if _ov[i, j]:
                merge_list.append([i, j])

    merged1 = np.zeros(len(objects1), dtype=np.bool)
    merged2 = np.zeros(len(objects2), dtype=np.bool)
    codes = []
    for i, j in merge_list:
        #print("merge")
        obj = random.choice([objects1[i], objects2[j]])
        codes.append(obj)
        merged1[i] = True
        merged2[j] = True

    for i, obj in enumerate(objects1):
        if not merged1[i]:
            codes.append(obj)

    for j, obj in enumerate(objects2):
        if not merged2[j]:
            codes.append(obj)


    

    return codes, rel_tran, rel_rot, affinity_info


def stitch_objects_affinity(objects1, objects2, rel_rot, rel_tran, mesh_dir=None):
    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    affinity_pred = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    has_matching = np.max(affinity_pred, axis=1) > 0.5
    matching = np.argmax(affinity_pred, axis=1)

    # adjust matching
    matching = list(matching)
    for i, j in enumerate(matching):
        if not has_matching[i]:
            continue
        if j == -1:
            continue
        boys = [i]
        for m, n in enumerate(matching):
            if i == m:
                continue
            if j == n:
                boys.append(m)
        boy_scores = affinity_pred[:, j][boys]
        boy_idx = boys[np.argmax(boy_scores)]
        
        for boy in boys:
            if boy != boy_idx:
                matching[boy] = -1
                has_matching[boy] = False

    #pdb.set_trace()

    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rel_rot)
    merged = np.zeros(len(objects2), dtype=np.bool)

    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran

        if has_matching[i]:
            j = matching[i]
            if merged[j]:
                continue
            mobj = objects2[j]
            merged[j] = True
            obj = merge_codes(obj, mobj, 'avg')

        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    affinity_info = {}
    affinity_info['hit_gt_match'] = None
    affinity_info['gt_match_in_proposal'] = None
    affinity_info['affinity_pred'] = None
    affinity_info['affinity_gt'] = None
    affinity_info['matching'] = None
    
    return codes, rel_tran, rel_rot, affinity_info


def stitch_objects_af_chamfer(objects1, objects2, rot_logits, tran_logits, mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False):
    # TOP-k sample
    topk_match = 2
    topk_rot = 5
    topk_tran = 10
    upperbound_match = 32

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]

    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    matching_list = get_matching_list(affinity_m, affinity_idx, len(objects1))

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    matching_proposals = itertools.product(*matching_list)
    matching_proposals = list([m for m in matching_proposals])
    if len(matching_proposals) > upperbound_match:
        #matching_proposals = random.sample(matching_proposals, upperbound_match)
        matching_proposals = matching_proposals[:upperbound_match]
    tqdm.write(str(matching_list))
    # pick up the best combination of (rot, tran, matching)
    min_dis = float('inf')
    best_comb = None
    gt_match_in_proposal = False
    for matching in (matching_proposals):
        matching = list(matching)
        # adjust matching
        for i, j in enumerate(matching):
            if j == -1:
                continue
            boys = [i]
            for m, n in enumerate(matching):
                if i == m:
                    continue
                if j == n:
                    boys.append(m)
            boy_scores = affinity_m[:, j][boys]
            boy_idx = boys[np.argmax(boy_scores)]
            
            for boy in boys:
                if boy != boy_idx:
                    matching[boy] = -1
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i]))
            corresponding_right_objs.append(object_to_pcd(objects2[j]))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        for rot_class in rot_proposals:
            for tran_class in tran_proposals:
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
        assert(len(transformation_proposals) == 50)

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    raise RuntimeError('pcd1.shape[0] == 0 or pcd2.shape[0] == 0:')
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != 50:
                raise RuntimeError('len(cd_pair) != 50')
        if len(cd_pairs) == 0: # no matching
            continue
        best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        if dis < min_dis:
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_dis = dis
            best_comb = (rot, tran, matching)
    # stitch accordingly
    codes = []
    rot, rel_tran, matching = best_comb
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    #rot = class2quat(rot_class)
    #rel_tran = class2tran(tran_class)
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            # average everything
            obj['shape'] = obj['shape'] / 2 + mobj['shape'] / 2
            obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
            obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
            obj['quat'] = random.choice([obj['quat'], mobj['quat']])
            obj['cmap'] = 'green'
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info


def stitch_objects_af_matching(objects1, objects2, rel_rot, rel_tran, mesh_dir=None, gt_affinity_m=None):
    """
    Stitch objects using affinity matrix and gt camera pose.
    """
    # TOP-k sample
    topk_match = 2
    # topk_rot = 5
    # topk_tran = 5
    upperbound_match = 2000

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if gt_affinity_m is None:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
     # top k
    matching_list = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= 0.5:
                continue
            options.append(j)
        if len(options) == 0:
            options.append(-1)
        matching_list.append(options)
    matching_proposals = itertools.product(*matching_list)
    matching_proposals = list([m for m in matching_proposals])
    if len(matching_proposals) > upperbound_match:
        matching_proposals = random.sample(matching_proposals, upperbound_match)
    tqdm.write(str(matching_list))
    
    # pick up the best combination of (matching)
    min_dis = float('inf')
    best_comb = None
    rot = rel_rot
    tran = rel_tran
    rot_quat = Quaternion(rot)
    for matching in matching_proposals:
        matching = list(matching)
        # adjust matching
        for i, j in enumerate(matching):
            if j == -1:
                continue
            boys = [i]
            for m, n in enumerate(matching):
                if i == m:
                    continue
                if j == n:
                    boys.append(m)
            boy_scores = affinity_m[:, j][boys]
            boy_idx = boys[np.argmax(boy_scores)]
            
            for boy in boys:
                if boy != boy_idx:
                    matching[i] = -1
                
        
        dis = 0
        diss = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            transformed_obj1 = deepcopy(objects1[i])
            quat = rot_quat * Quaternion(transformed_obj1['quat'])
            transformed_obj1['quat'] = quat.elements
            transformed_obj1['trans'] = rot_quat.rotate(transformed_obj1['trans']) + tran
            pcd1 = object_to_pcd(transformed_obj1)
            pcd2 = object_to_pcd(objects2[j])
            if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                diss.append(400.0)
                continue
            #dis += chamfer_distance(pcd1, pcd2)
            diss.append(chamfer_distance(pcd1, pcd2))

        dis = sum(diss)
        if dis < min_dis:
            min_dis = dis
            best_comb = (rot, tran, matching)

    # stitch accordingly
    codes = []
    rot, rel_tran, matching = best_comb
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran

        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            # average everything
            obj['shape'] = obj['shape'] / 2 + mobj['shape'] / 2
            obj['trans'] = obj['trans'] / 2 + mobj['trans'] / 2
            obj['scale'] = obj['scale'] / 2 + mobj['scale'] / 2
            obj['quat'] = random.choice([obj['quat'], mobj['quat']])
            # obj['quat'] = mean_quaternion(obj['quat'], mobj['quat'])
            
            # set up mixed colormap
            obj['cmap'] = 'green'

        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes


def stitch_objects_rand_chamfer(objects1, objects2, rot_logits, tran_logits, mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False):
    # lambdas
    lambda_nomatch = 1
    
    # TOP-k sample
    topk_match = 2
    topk_rot = 5
    topk_tran = 10
    upperbound_match = 32

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= 0.5:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch])
    
    # pick up the best combination of (rot, tran, matching)
    min_loss = float('inf')
    best_comb = None
    for matching, num_nomatch in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i]))
            corresponding_right_objs.append(object_to_pcd(objects2[j]))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        for rot_class in rot_proposals:
            for tran_class in tran_proposals:
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
        assert(len(transformation_proposals) == 50)

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != 50:
                raise RuntimeError('len(cd_pair) != 50:')
                pass
        if len(cd_pairs) == 0: # no matching
            continue
        best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        loss = dis + lambda_nomatch * num_nomatch + lambda_1 * tran_logits
        if loss < min_loss:
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_loss = loss
            best_comb = (rot, tran, matching)
    
    if best_comb is None:
        # if there is no matching at all, we should trust relative pose.
        matching = [-1 for _ in range(len(objects1))]
        rot_class = rot_logits.argmax()
        tran_class = tran_logits.argmax()
        rot = class2quat(rot_class)
        rel_tran = class2tran(tran_class)
    else:
        rot, rel_tran, matching = best_comb

    # collect affinity info
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    affinity_info['affinity_pred'] = affinity_m
    affinity_info['affinity_gt'] = gt_affinity_m
    affinity_info['matching'] = matching
    
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    
    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            obj = merge_codes(obj, mobj, 'avg')
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info


def stitch_objects_rand_chamfer_prob(objects1, objects2, rot_logits, tran_logits, mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False):
    # lambdas
    lambda_nomatch = 3
    lambda_rots = 5
    lambda_trans = 1
    lambda_af = 10
    
    # TOP-k sample
    topk_match = 2
    topk_rot = 3
    topk_tran = 10
    upperbound_match = 128
    #thres = 0.75
    thres = 0.5

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= thres:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    rot_proposals_prob = rot_logits[rot_proposals]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    tran_proposals_prob = tran_logits[tran_proposals]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # compute scores
        scores = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            scores.append(affinity_m[i, j])
        scores = np.array(scores)

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch, scores])
    
    # pick up the best combination of (rot, tran, matching)
    min_loss = float('inf')
    best_comb = None
    for matching, num_nomatch, scores in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i]))
            corresponding_right_objs.append(object_to_pcd(objects2[j]))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        transformation_probs = {'tran':[], 'rot':[]}
        for rot_class, rot_prob in zip(rot_proposals, rot_proposals_prob):
            for tran_class, tran_prob in zip(tran_proposals, tran_proposals_prob):
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
                transformation_probs['tran'].append(tran_prob)
                transformation_probs['rot'].append(rot_prob)
        for key in transformation_probs.keys():
            transformation_probs[key] = np.array(transformation_probs[key])
        assert(len(transformation_proposals) == (topk_rot * topk_tran))

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != (topk_rot * topk_tran):
                raise RuntimeError('len(cd_pair) != topk_rot * topk_tran:')
                pass
        if len(cd_pairs) == 0: # no matching
            continue
        #best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        cd_cost = chamfer_metric(cd_pairs, metric='avg')
        assert(len(cd_cost) == len(transformation_proposals))

        losses = cd_cost + lambda_nomatch * num_nomatch \
                + lambda_trans * (1 - transformation_probs['tran']) \
                + lambda_rots * (1 - transformation_probs['rot']) \
                + lambda_af * (1 - scores).mean()
        #loss = dis + lambda_nomatch * num_nomatch
        loss = losses.min()
        if loss < min_loss:
            best_pair_id = losses.argmin()
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_loss = loss
            best_comb = (rot, tran, matching)
    
    if best_comb is None:
        # if there is no matching at all, we should trust relative pose.
        matching = [-1 for _ in range(len(objects1))]
        rot_class = rot_logits.argmax()
        tran_class = tran_logits.argmax()
        rot = class2quat(rot_class)
        rel_tran = class2tran(tran_class)
    else:
        rot, rel_tran, matching = best_comb

    # collect affinity info
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    affinity_info['affinity_pred'] = affinity_m
    affinity_info['affinity_gt'] = gt_affinity_m
    affinity_info['matching'] = matching
    
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    
    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            obj = merge_codes(obj, mobj, 'avg')
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info


def stitch_objects_asso_shape(objects1, objects2, rot_logits, tran_logits, tester, mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False):
    """
    stitch_objects_rand_chamfer_prob + average shape space
    """
    # lambdas
    lambda_nomatch = 1
    lambda_rots = 5
    lambda_trans = 1
    lambda_af = 5
    
    # TOP-k sample
    topk_match = 2
    topk_rot = 3
    topk_tran = 10
    upperbound_match = 64
    #thres = 0.75
    thres = 0.5

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= thres:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    rot_proposals_prob = rot_logits[rot_proposals]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    tran_proposals_prob = tran_logits[tran_proposals]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # compute scores
        scores = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            scores.append(affinity_m[i, j])
        scores = np.array(scores)

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch, scores])
    
    # pick up the best combination of (rot, tran, matching)
    min_loss = float('inf')
    best_comb = None
    for matching, num_nomatch, scores in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i]))
            corresponding_right_objs.append(object_to_pcd(objects2[j]))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        transformation_probs = {'tran':[], 'rot':[]}
        for rot_class, rot_prob in zip(rot_proposals, rot_proposals_prob):
            for tran_class, tran_prob in zip(tran_proposals, tran_proposals_prob):
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
                transformation_probs['tran'].append(tran_prob)
                transformation_probs['rot'].append(rot_prob)
        for key in transformation_probs.keys():
            transformation_probs[key] = np.array(transformation_probs[key])
        assert(len(transformation_proposals) == (topk_rot * topk_tran))

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != (topk_rot * topk_tran):
                raise RuntimeError('len(cd_pair) != topk_rot * topk_tran:')
                pass
        if len(cd_pairs) == 0: # no matching
            continue
        #best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        cd_cost = chamfer_metric(cd_pairs, metric='avg')
        assert(len(cd_cost) == len(transformation_proposals))

        losses = cd_cost + lambda_nomatch * num_nomatch \
                + lambda_trans * (1 - transformation_probs['tran']) \
                + lambda_rots * (1 - transformation_probs['rot']) \
                + lambda_af * (1 - scores).mean()
        #loss = dis + lambda_nomatch * num_nomatch
        loss = losses.min()
        if loss < min_loss:
            best_pair_id = losses.argmin()
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_loss = loss
            best_comb = (rot, tran, matching)
    
    if best_comb is None:
        # if there is no matching at all, we should trust relative pose.
        matching = [-1 for _ in range(len(objects1))]
        rot_class = rot_logits.argmax()
        tran_class = tran_logits.argmax()
        rot = class2quat(rot_class)
        rel_tran = class2tran(tran_class)
    else:
        rot, rel_tran, matching = best_comb

    # collect affinity info
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    affinity_info['affinity_pred'] = affinity_m
    affinity_info['affinity_gt'] = gt_affinity_m
    affinity_info['matching'] = matching
    
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    
    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            obj = merge_codes(obj, mobj, 'avg_shape', tester=tester)
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info


def stitch_objects_rand_chamfer_ambiguity(objects1, objects2, rot_logits, tran_logits, mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False):
    # lambdas
    lambda_nomatch = 1
    lambda_rots = 5
    lambda_trans = 1
    lambda_af = 5
    
    # TOP-k sample
    topk_match = 2
    topk_rot = 3
    topk_tran = 10
    upperbound_match = 64
    thres = 0.5

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= thres:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    rot_proposals_prob = rot_logits[rot_proposals]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    tran_proposals_prob = tran_logits[tran_proposals]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # compute scores
        scores = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            scores.append(affinity_m[i, j])
        scores = np.array(scores)

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch, scores])
    
    # pick up the best combination of (rot, tran, matching)
    #min_loss = float('inf')
    #best_comb = None
    combs = []
    for matching, num_nomatch, scores in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i]))
            corresponding_right_objs.append(object_to_pcd(objects2[j]))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        transformation_probs = {'tran':[], 'rot':[]}
        for rot_class, rot_prob in zip(rot_proposals, rot_proposals_prob):
            for tran_class, tran_prob in zip(tran_proposals, tran_proposals_prob):
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
                transformation_probs['tran'].append(tran_prob)
                transformation_probs['rot'].append(rot_prob)
        for key in transformation_probs.keys():
            transformation_probs[key] = np.array(transformation_probs[key])
        assert(len(transformation_proposals) == (topk_rot * topk_tran))

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != (topk_rot * topk_tran):
                raise RuntimeError('len(cd_pair) != topk_rot * topk_tran:')
                pass
        if len(cd_pairs) == 0: # no matching
            continue
        #best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        cd_cost = chamfer_metric(cd_pairs, metric='avg')
        assert(len(cd_cost) == len(transformation_proposals))

        losses = cd_cost + lambda_nomatch * num_nomatch \
                + lambda_trans * (1 - transformation_probs['tran']) \
                + lambda_rots * (1 - transformation_probs['rot']) \
                + lambda_af * (1 - scores).mean()

        for i, loss in enumerate(losses):
            comb = {
                'tran': transformation_proposals[i]['tran'],
                'rot': transformation_proposals[i]['rot'],
                'loss': loss,
                'matching': matching
            }
            combs.append(comb)

        #loss = losses.min()
        #if loss < min_loss:
        #    best_pair_id = losses.argmin()
        #    tran = transformation_proposals[best_pair_id]['tran']
        #    rot = transformation_proposals[best_pair_id]['rot']
        #    min_loss = loss
        #    best_comb = (rot, tran, matching)

    combs.sort(key=lambda comb: comb['loss'])
    final_combs = []
    cur_loss = -1
    for comb in combs:
        if comb['loss'] - cur_loss > 1e-1:
            final_combs.append(comb)
        if len(final_combs) >= 10:
            break

    codes_options = []
    for comb in final_combs:
        rot = comb['rot']
        rel_tran = comb['tran']
        matching = comb['matching']
        
        # stitch accordingly
        codes = []
        rel_quat = Quaternion(rot)
        merged = np.zeros(len(objects2), dtype=np.bool)
        for i, obj_cp in enumerate(objects1):
            obj = deepcopy(obj_cp)
            for k in obj.keys():
                if k == 'quat':
                    try:
                        quat = rel_quat * Quaternion(obj[k])
                    except ValueError:
                        quat = rel_quat * Quaternion(obj[k][0])
                    obj[k] = quat.elements
                elif k == 'trans':
                    obj[k] = rel_quat.rotate(obj[k]) + rel_tran
            if matching[i] != -1:
                j = matching[i]
                mobj = deepcopy(objects2[j])
                merged[j] = True
                obj = merge_codes(obj, mobj, 'avg')
            codes.append(obj)
            
        # add unmerged objs in objects2 to codes
        for j, obj in enumerate(objects2):
            if not merged[j]:
                codes.append(deepcopy(obj))

        codes_options.append(codes)
    
    return codes_options


def generate_volume_trans(objects1, objects2, rel_rot, rel_tran, mesh_dir=None):
    results = {
        'view1': [],
        'view2': []
    }
    codes1 = suncg_parse.convert_codes_list_to_old_format(objects1)
    codes2 = suncg_parse.convert_codes_list_to_old_format(objects2)
    for obj in codes1:
        volume, transform = render_utils.prediction_to_entity(obj)
        result = {
            'volume': volume,
            'transform': transform,
        }
        results['view1'].append(result)
    for obj in codes2:
        volume, transform = render_utils.prediction_to_entity(obj)
        result = {
            'volume': volume,
            'transform': transform,
        }
        results['view2'].append(result)
    with open(os.path.join(mesh_dir, 'output.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return []


def stitch_objects_edge(objects1, objects2, rot_logits, tran_logits, tester, 
        mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False, opts=None):
    # lambdas
    if opts is None:
        lambda_nomatch = 1
        lambda_rots = 5
        lambda_trans = 1
        lambda_af = 5
    else:
        lambda_nomatch = opts.lambda_nomatch
        lambda_rots = opts.lambda_rots
        lambda_trans = opts.lambda_trans
        lambda_af = opts.lambda_af
    
    #print(lambda_nomatch)
    #print(lambda_rots)
    #print(lambda_trans)
    #print(lambda_af)

    # TOP-k sample
    topk_match = 3
    topk_rot = 3
    topk_tran = 10
    upperbound_match = 128
    #thres = 0.75
    #thres = 0.79
    thres = 0.5

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= thres:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    rot_proposals_prob = rot_logits[rot_proposals]
    tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    tran_proposals_prob = tran_logits[tran_proposals]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # compute scores
        scores = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            scores.append(affinity_m[i, j])
        scores = np.array(scores)

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch, scores])
    
    # pick up the best combination of (rot, tran, matching)
    min_loss = float('inf')
    best_comb = None
    for matching, num_nomatch, scores in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i], edge=True))
            corresponding_right_objs.append(object_to_pcd(objects2[j], edge=True))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        transformation_probs = {'tran':[], 'rot':[]}
        for rot_class, rot_prob in zip(rot_proposals, rot_proposals_prob):
            for tran_class, tran_prob in zip(tran_proposals, tran_proposals_prob):
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
                transformation_probs['tran'].append(tran_prob)
                transformation_probs['rot'].append(rot_prob)
        for key in transformation_probs.keys():
            transformation_probs[key] = np.array(transformation_probs[key])
        assert(len(transformation_proposals) == (topk_rot * topk_tran))

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            if len(cd_pair) != (topk_rot * topk_tran):
                raise RuntimeError('len(cd_pair) != topk_rot * topk_tran:')
                pass
        if len(cd_pairs) == 0: # no matching
            continue
        #best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        cd_cost = chamfer_metric(cd_pairs, metric='avg')
        assert(len(cd_cost) == len(transformation_proposals))

        losses = cd_cost + lambda_nomatch * num_nomatch \
                + lambda_trans * (1 - transformation_probs['tran']) \
                + lambda_rots * (1 - transformation_probs['rot']) \
                + lambda_af * (1 - scores).mean()
        #loss = dis + lambda_nomatch * num_nomatch
        loss = losses.min()
        if loss < min_loss:
            best_pair_id = losses.argmin()
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_loss = loss
            best_comb = (rot, tran, matching)
    
    if best_comb is None:
        # if there is no matching at all, we should trust relative pose.
        matching = [-1 for _ in range(len(objects1))]
        rot_class = rot_logits.argmax()
        tran_class = tran_logits.argmax()
        rot = class2quat(rot_class)
        rel_tran = class2tran(tran_class)
    else:
        rot, rel_tran, matching = best_comb

    # collect affinity info
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    affinity_info['affinity_pred'] = affinity_m
    affinity_info['affinity_gt'] = gt_affinity_m
    affinity_info['matching'] = matching
    
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    
    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            #obj = merge_codes(obj, mobj, 'avg', tester=tester)
            obj = merge_codes(obj, mobj, 'avg_voxel', tester=tester)
            #obj = merge_codes(obj, mobj, 'avg_shape', tester=tester)
            #obj = merge_codes(obj, mobj, 'avg_rot', tester=tester)
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info


def stitch_objects_edge_wpose(objects1, objects2, rel_rot, rel_tran, tester, 
        mesh_dir=None, gt_affinity_m=None, use_gt_affinity=False, opts=None):
    """
    The same as stitch_objects_edge but use input relative camera pose
    """

    # lambdas
    if opts is None:
        lambda_nomatch = 1
        lambda_rots = 5
        lambda_trans = 1
        lambda_af = 5
    else:
        lambda_nomatch = opts.lambda_nomatch
        lambda_rots = opts.lambda_rots
        lambda_trans = opts.lambda_trans
        lambda_af = opts.lambda_af

    # TOP-k sample
    topk_match = 3
    topk_rot = 3
    topk_tran = 10
    upperbound_match = 128
    #thres = 0.75
    #thres = 0.79
    thres = 0.5

    # collate id
    id1 = np.zeros((len(objects1), opts.nz_id))
    for idx, obj in enumerate(objects1):
        id1[idx] = obj['id']
    id2 = np.zeros((len(objects2), opts.nz_id))
    for idx, obj in enumerate(objects2):
        id2[idx] = obj['id']
    id1 = torch.FloatTensor(id1)
    id2 = torch.FloatTensor(id2)

    # calculate affinity matrix and matching
    if not use_gt_affinity:
        affinity_m = forward_affinity(id1, id2, mesh_dir=mesh_dir)
    else:
        affinity_m = gt_affinity_m
    affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]
    gt_matching_idx = np.argsort(gt_affinity_m, axis=1)[:, ::-1]
    gt_matching_list = get_matching_list(gt_affinity_m, gt_matching_idx, len(objects1))
    
    af_options = []
    for i in range(len(objects1)):
        options = []
        for j in affinity_idx[i]:
            if affinity_m[i][j] <= thres:
                continue
            options.append(j)
        af_options.append(options)

    # top k
    #rot_proposals = np.argsort(rot_logits)[::-1][:topk_rot]
    #rot_proposals_prob = rot_logits[rot_proposals]
    #tran_proposals = np.argsort(tran_logits)[::-1][:topk_tran]
    #tran_proposals_prob = tran_logits[tran_proposals]
    matching_proposals = []
    gt_match_in_proposal = False
    for _ in range(upperbound_match):
        matching = []
        num_nomatch = 0
        for i in range(len(objects1)):
            options = []
            for op in af_options[i]:
                if op not in matching:
                    options.append(int(op))
            options.append(-1)
            m = random.choice(options)
            if m == -1:
                num_nomatch += 1
            matching.append(m)

        # check matching is valid
        if not is_valid_matching(matching):
            raise RuntimeError('invalid matching')

        # compute scores
        scores = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            scores.append(affinity_m[i, j])
        scores = np.array(scores)

        # check if gt_matching_list in matching_proposals
        if (matching == np.array(gt_matching_list).flatten()).all():
            gt_match_in_proposal = True
        matching_proposals.append([matching, num_nomatch, scores])
    
    # pick up the best combination of (rot, tran, matching)
    min_loss = float('inf')
    best_comb = None
    for matching, num_nomatch, scores in (matching_proposals):
        # for each matching
        left_objs = []
        corresponding_right_objs = []
        for i, j in enumerate(matching):
            if j == -1:
                continue
            left_objs.append(object_to_pcd(objects1[i], edge=True))
            corresponding_right_objs.append(object_to_pcd(objects2[j], edge=True))
        assert(len(left_objs) == len(corresponding_right_objs))
        transformation_proposals = []
        transformation_probs = {'tran':[], 'rot':[]}
        """
        for rot_class, rot_prob in zip(rot_proposals, rot_proposals_prob):
            for tran_class, tran_prob in zip(tran_proposals, tran_proposals_prob):
                rot = class2quat(rot_class)
                tran = class2tran(tran_class)
                transformation_proposals.append({'tran': tran, 'rot': rot})
                transformation_probs['tran'].append(tran_prob)
                transformation_probs['rot'].append(rot_prob)
        """
        transformation_proposals.append({'tran': rel_tran, 'rot': rel_rot})
        transformation_probs['tran'].append(1.0)
        transformation_probs['rot'].append(1.0)


        for key in transformation_probs.keys():
            transformation_probs[key] = np.array(transformation_probs[key])
        #assert(len(transformation_proposals) == (topk_rot * topk_tran))

        cd_pairs = []
        for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
            pcd1s = []
            pcd2s = []
            for transformation_proposal in transformation_proposals:
                rot = transformation_proposal['rot']
                tran = transformation_proposal['tran']
                rot_quat = Quaternion(rot).rotation_matrix
                transformed_obj1 = (rot_quat@pcd1.T).T+tran
                if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
                    pcd1s.append([[1000,1000,1000]])
                    pcd2s.append([[-1000,-1000,-1000]])
                    continue
                pcd1s.append(transformed_obj1)
                pcd2s.append(pcd2)
            cd_pair = chamfer_distance(pcd1s, pcd2s)
            cd_pairs.append(cd_pair.flatten())
            #if len(cd_pair) != (topk_rot * topk_tran):
            #    raise RuntimeError('len(cd_pair) != topk_rot * topk_tran:')
            #    pass
        if len(cd_pairs) == 0: # no matching
            continue
        #best_pair_id, dis = chamfer_metric(cd_pairs, metric="avg")
        cd_cost = chamfer_metric(cd_pairs, metric='avg')
        assert(len(cd_cost) == len(transformation_proposals))

        losses = cd_cost + lambda_nomatch * num_nomatch \
                + lambda_trans * (1 - transformation_probs['tran']) \
                + lambda_rots * (1 - transformation_probs['rot']) \
                + lambda_af * (1 - scores).mean()
        #loss = dis + lambda_nomatch * num_nomatch
        loss = losses.min()
        if loss < min_loss:
            best_pair_id = losses.argmin()
            tran = transformation_proposals[best_pair_id]['tran']
            rot = transformation_proposals[best_pair_id]['rot']
            min_loss = loss
            best_comb = (rot, tran, matching)
    
    if best_comb is None:
        # if there is no matching at all, we should trust relative pose.
        matching = [-1 for _ in range(len(objects1))]
        #rot_class = rot_logits.argmax()
        #tran_class = tran_logits.argmax()
        #rot = class2quat(rot_class)
        #rel_tran = class2tran(tran_class)
        rot = rel_rot
        rel_tran = rel_tran
    else:
        rot, rel_tran, matching = best_comb

    # collect affinity info
    affinity_info = {}
    affinity_info['hit_gt_match'] = (matching == np.array(gt_matching_list).flatten()).all()
    affinity_info['gt_match_in_proposal'] = gt_match_in_proposal
    affinity_info['affinity_pred'] = affinity_m
    affinity_info['affinity_gt'] = gt_affinity_m
    affinity_info['matching'] = matching
    
    save_affinity_after_stitch(affinity_m, len(objects1), len(objects2), matching, mesh_dir)
    
    # stitch accordingly
    codes = []
    rel_quat = Quaternion(rot)
    merged = np.zeros(len(objects2), dtype=np.bool)
    for i, obj in enumerate(objects1):
        for k in obj.keys():
            if k == 'quat':
                try:
                    quat = rel_quat * Quaternion(obj[k])
                except ValueError:
                    quat = rel_quat * Quaternion(obj[k][0])
                obj[k] = quat.elements
            elif k == 'trans':
                obj[k] = rel_quat.rotate(obj[k]) + rel_tran
        if matching[i] != -1:
            j = matching[i]
            mobj = objects2[j]
            merged[j] = True
            #obj = merge_codes(obj, mobj, 'avg', tester=tester)
            obj = merge_codes(obj, mobj, 'avg_voxel', tester=tester)
            #obj = merge_codes(obj, mobj, 'avg_shape', tester=tester)
            #obj = merge_codes(obj, mobj, 'avg_rot', tester=tester)
        codes.append(obj)
        
    # add unmerged objs in objects2 to codes
    for j, obj in enumerate(objects2):
        if not merged[j]:
            codes.append(obj)

    return codes, rel_tran, rot, affinity_info
