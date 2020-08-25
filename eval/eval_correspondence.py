"""
Evaluation using detection.
"""
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import pdb
import os
from sklearn.metrics import average_precision_score, roc_auc_score
#from sklearn.metrics import plot_precision_recall_curve

parser = argparse.ArgumentParser()
parser.add_argument('--split_file', type=str, 
    default='../script/v3_rpnet_split_relative3d/validation_set.txt', 
    help='split file path')
parser.add_argument('--result_dir', type=str, default='/x/syqian/mesh_output', help='results directory')
args = parser.parse_args()

def adjust_matching(matching, affinity_m):
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

    return matching

def acc_bench(affinity_gt, matching_pred, tp, fp, tn, fn):
    num_obj1 = affinity_gt.shape[0]
    num_obj2 = affinity_gt.shape[1]
    for i in range(num_obj1):
        for j in range(num_obj2):
            if affinity_gt[i, j] < 0.5:
                if matching_pred[i] != j:
                    tn += 1
                else:
                    fp += 1
            else:
                if matching_pred[i] == j:
                    tp += 1
                else:
                    fn += 1
    return tp, fp, tn, fn

def acc_ap(affinity_gt, affinity_pred, matching_pred, y_true, y_score):
    num_obj1 = affinity_gt.shape[0]
    num_obj2 = affinity_gt.shape[1]
    for i in range(num_obj1):
        for j in range(num_obj2):
            score = affinity_pred[i, j]
            if affinity_gt[i, j] < 0.5:
                y_true.append(0)
            else:
                y_true.append(1)

            if matching_pred is None:
                y_score.append(score)
                continue

            if matching_pred[i] == j:
                y_score.append(score)
            else:
                y_score.append(score / 2)


def report_ap(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    #print(y_true)
    #print(y_score)
    #pdb.set_trace()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print("\t auc: {}".format(auc))
    print("\t ap: {}".format(ap))
    #disp = plot_precision_recall_curve(classifier, X_test, y_test)


def report_metric(tp, fp, tn, fn):
    print('\t tp: {}'.format(tp))
    print('\t fp: {}'.format(fp))
    print('\t tn: {}'.format(tn))
    print('\t fn: {}'.format(fn))
    acc = (tp + tn) / (tp + fp + tn + fn)
    print('\t accuracy: {}'.format(acc))
    print('\t sensitivity: {}'.format(tp / (tp + fn + 1e-3)))
    print('\t specificity: {}'.format(tn / (fp + tn + 1e-3)))
    print('\t precision: {}'.format(tp / (tp + fp + 1e-3)))
    print('\t recall: {}'.format(tp / (tp + fn + 1e-3)))


def main():
    f = open(args.split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    #lines = lines[:100]

    bb = [0, 0, 0, 0]
    bb_top1 = [0, 0, 0, 0]
    bb_allneg = [0, 0, 0, 0]

    neg_y_true, neg_y_score = [], []
    aff_y_true, aff_y_score = [], []
    top1_y_true, top1_y_score = [], []
    ours_y_true, ours_y_score = [], []

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
        result_dir = os.path.join(args.result_dir, pair_id)
        rf = open(os.path.join(result_dir, 'results.pkl'), 'rb')
        results = pickle.load(rf)
        rf.close()
        affinity_pred = results['affinity_pred']
        affinity_gt = results['affinity_gt']
        af_m = affinity_pred > 1e-7
        num_obj1 = af_m[:, 0].sum()
        num_obj2 = af_m[0].sum()
        if num_obj1 == 0 or num_obj2 == 0:
            continue
        affinity_pred = affinity_pred[:num_obj1, :num_obj2]
        affinity_gt = affinity_gt[:num_obj1, :num_obj2]
        matching_pred = np.array(results['matching_pred'])
        matching_pred_top1 = affinity_pred.argmax(axis=1)
        matching_pred_top1[affinity_pred.max(axis=1) < 0.5] = -1
        matching_pred_top1 = adjust_matching(matching_pred_top1, affinity_pred)

        # tp, fp, tn, fn
        """
        bb = acc_bench(affinity_gt, matching_pred, *bb)
        bb_top1 = acc_bench(affinity_gt, matching_pred_top1, *bb_top1)
        matching_pred_allneg = - np.ones(matching_pred.shape[0])
        bb_allneg = acc_bench(affinity_gt, matching_pred_allneg, *bb_allneg)
        """

        # ap
        matching_pred_allneg = - np.ones(matching_pred.shape[0])
        acc_ap(affinity_gt, np.zeros_like(affinity_gt), matching_pred_allneg, neg_y_true, neg_y_score)
        acc_ap(affinity_gt, affinity_pred, None, aff_y_true, aff_y_score)
        acc_ap(affinity_gt, affinity_pred, matching_pred_top1, top1_y_true, top1_y_score)
        acc_ap(affinity_gt, affinity_pred, matching_pred, ours_y_true, ours_y_score)


    # tp, fp, tn, fn
    """
    print("all neg:")
    report_metric(*bb_allneg)
    print("top1:")
    report_metric(*bb_top1)
    print("stitching:")
    report_metric(*bb)
    """
    
    print("all neg:")
    report_ap(neg_y_true, neg_y_score)
    print("aff:")
    report_ap(aff_y_true, aff_y_score)
    print("top1:")
    report_ap(top1_y_true, top1_y_score)
    print("stitching:")
    report_ap(ours_y_true, ours_y_score)



if __name__=='__main__':
    main()
