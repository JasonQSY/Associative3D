import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
import pdb
import os
import transformations
import math 
import seaborn as sns

sns.set()
#plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument('--split_file', type=str, 
    default='../script/v3_rpnet_split_relative3d/validation_set.txt', 
    help='split file path')
parser.add_argument('--result_dir', type=str, default='/x/syqian/gtbbox_rand_chamfer_rot5_trans10_v3_validation/', help='results directory')
args = parser.parse_args()

def plot_err_vs_fraction(data, ran, label, title):
    data = np.array(data)
    data.sort()
    total_len = len(data)
    pair = []
    for idx, d in enumerate(data):
        pair.append([idx/total_len, d])
    pair = np.array(pair)
    #plt.plot(pair[:,0], pair[:,1], label=label)
    figure = plt.figure(figsize=(100, 100))
    sns.lineplot(pair[:,0], pair[:,1], label=label)
    plt.title(title)
    plt.xlabel("Fraction of Data")
    plt.ylabel("Error")
    plt.ylim((0, ran))
    plt.xlim((0, 1))
    figure.close()
    
        
def getErrTran(gt, pred):
    return np.linalg.norm(pred-gt, ord=2)


def quat_dist(gt, pred):
    try:
        rot_pred = transformations.quaternion_matrix(pred.numpy())
        rot_gt = transformations.quaternion_matrix(gt.numpy())
    except AttributeError:
        rot_pred = transformations.quaternion_matrix(pred)
        rot_gt = transformations.quaternion_matrix(gt)

    rot_rel = np.matmul(rot_pred, np.transpose(rot_gt))
    quat_rel = transformations.quaternion_from_matrix(rot_rel, isprecise=True)
    try:
        quat_rel = np.clip(quat_rel, a_min=-1, a_max=1)
        angle = math.acos(abs(quat_rel[0]))*360/math.pi
    except ValueError:
        pdb.set_trace()
    return angle


def getErrRot(gt, pred):
    d = np.array(np.abs(np.sum(np.multiply(pred, gt))))
    if d > 1:
        d = 1
    if d < -1:
        d = -1
    return np.arccos(d) * 180/math.pi * 2


def plot_performance(top1, after_stitch):
    plot_err_vs_fraction(top1['tran'], 10, "top1_tran", title='Translation Analysis')
    plot_err_vs_fraction(after_stitch['tran'], 10, "after_stitch_tran", title='Translation Analysis')
    plt.legend()
    plt.savefig('vis/tran.png')
    plt.close()
    plot_err_vs_fraction(top1['rot'], 180, "top1_rot", title='Rotation Analysis')
    plot_err_vs_fraction(after_stitch['rot'], 180, "after_stitch_rot", title='Rotation Analysis')
    plt.legend()
    plt.savefig('vis/rot.png')
    plt.close()


def print_in_latex_format(top1_err_trans, top1_err_rots, after_stitch_trans, after_stitch_rots, tran_threshold=1, rot_threshold=30):
    top1_count_tran = 0
    top1_count_rot = 0
    for err_tran in top1_err_trans:
        if err_tran <= tran_threshold:
            top1_count_tran += 1
    for err_rot in top1_err_rots:
        if err_rot <= rot_threshold:
            top1_count_rot += 1


    after_stitch_count_tran = 0
    after_stitch_count_rot = 0
    for err_tran in after_stitch_trans:
        if err_tran <= tran_threshold:
            after_stitch_count_tran += 1
    for err_rot in after_stitch_rots:
        if err_rot <= rot_threshold:
            after_stitch_count_rot += 1
    print("\\begin{tabular} {c|ccc|ccc }")
    print("\\toprule")
    print("             &           & Translation (meters)  &                   &           & Rotation  (degrees)   &                           \\\\")
    print("Method       & Median    & Mean                  & (Err $\leq$ {}m)\%    & Median    & Mean                  & (Err $\\leq$ {}".format(tran_threshold, rot_threshold) + "$^{ \\circ }$)\%   \\\\")
    print("\\midrule")
    print(" RP Module   & {:.2f}     & {:.2f}                 & {:.2f}             & {:.2f}     & {:.2f}                 & {:.2f}                     \\\\".format(
        np.median(top1_err_trans), np.mean(top1_err_trans), 100*top1_count_tran/len(top1_err_trans), 
        np.median(top1_err_rots), np.mean(top1_err_rots), 100*top1_count_rot/len(top1_err_rots)))
    print(" + Stitching & {:.2f}     & {:.2f}                 & {:.2f}             & {:.2f}     & {:.2f}                 & {:.2f}                     \\\\".format(
        np.median(after_stitch_trans), np.mean(after_stitch_trans), 100*after_stitch_count_tran/len(after_stitch_trans), 
        np.median(after_stitch_rots), np.mean(after_stitch_rots), 100*after_stitch_count_rot/len(after_stitch_rots)))
    
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    f = open(args.split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]

    trans_errors = []
    rot_errors = []
    scale_errors = []
    top1_rel_pose_err = {'tran': [], 'rot': []}
    stitch_rel_pose_err = {'tran': [], 'rot': []}
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

        # read results
        result_dir = os.path.join(args.result_dir, pair_id)
        rf = open(os.path.join(result_dir, 'results.pkl'), 'rb')
        results = pickle.load(rf)
        # TOP1
        top1_rel_pose_err['tran'].append(getErrTran(results['relpose']['tran_gt'], results['relpose']['tran_pred']))
        rot_err_6 = quat_dist(results['relpose']['rot_gt'], results['relpose']['rot_pred'])
        rot_err_3 = getErrRot(results['relpose']['rot_gt'], results['relpose']['rot_pred'])
        if (np.abs(rot_err_6 - rot_err_3)>1e-1):
            import pdb;pdb.set_trace()
            pass
        top1_rel_pose_err['rot'].append(rot_err_6)
        # AFTER STITCH
        stitch_rel_pose_err['tran'].append(getErrTran(results['relpose']['tran_gt'], results['relpose_after_stitch']['translation']))
        rot_err_6 = quat_dist(results['relpose']['rot_gt'], results['relpose_after_stitch']['rotation'])
        rot_err_3 = getErrRot(results['relpose']['rot_gt'], results['relpose_after_stitch']['rotation'])
        if (np.abs(rot_err_6 - rot_err_3)>1e-1):
            import pdb;pdb.set_trace()
            pass
        stitch_rel_pose_err['rot'].append(rot_err_6)

    
    for key in top1_rel_pose_err.keys():
        top1_rel_pose_err[key] = np.array(top1_rel_pose_err[key])
    for key in stitch_rel_pose_err.keys():
        stitch_rel_pose_err[key] = np.array(stitch_rel_pose_err[key])
    
    #plot_performance(top1_rel_pose_err, stitch_rel_pose_err)
    print_in_latex_format(top1_err_trans=top1_rel_pose_err['tran'], top1_err_rots=top1_rel_pose_err['rot'], 
                            after_stitch_trans=stitch_rel_pose_err['tran'], after_stitch_rots=stitch_rel_pose_err['rot'])


if __name__ == '__main__':
    main()
