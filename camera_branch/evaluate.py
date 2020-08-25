import argparse
import json
import math
import numpy as np
import os
import pickle
import torch

from copy import deepcopy
from collections import namedtuple
from tqdm import tqdm
from torch import optim
from scipy.special import softmax

from network import CameraBranch
from suncg import get_dataloader_benchmark


def run(model, dataloader, device, flags, dataset):
    # ------- Items contained in summary.npy
    loss = [[], []]
    errors = [[], []]
    accuracy = [[], []]
    acc_threshold = [1.0, 30.0] # threshold for translation and rotation error to say prediction is correct.
    preds = {'tran':[], 'rot':[], 'tran_cls':[], 'rot_cls':[]}
    gts = {'tran':[], 'rot':[], 'tran_cls':[], 'rot_cls':[]}
    logits_sms = {'tran':[], 'rot':[]}
    # ------- END of items contained in summary.npy
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Prepare
            img1 = batch['img1']
            img2 = batch['img2']
            img1 = img1.to(device)
            img2 = img2.to(device)
            gt_relative_pose = batch['relative_pose']
            gt_relative_pose['tran'] = gt_relative_pose['tran'].to(device)
            gt_relative_pose['rot'] = gt_relative_pose['rot'].to(device)
            if flags.loss_fn == 'C':
                gt_tran_cls = batch['tran_cls']
                gt_tran_cls = gt_tran_cls.to(device)
                gt_rot_cls = batch['rot_cls']
                gt_rot_cls = gt_rot_cls.to(device)
            # Inference & Loss
            pred = model(img1, img2)
            if flags.loss_fn == 'R':
                _, tran_loss, rot_loss = model.module.regression_loss(pred, gt_relative_pose)
            elif flags.loss_fn == 'C':
                _, tran_loss, rot_loss = model.module.classification_loss(pred, gt_tran_cls, gt_rot_cls)
            else:
                raise NotImplementedError
            tran_loss = tran_loss.item()
            rot_loss = rot_loss.item()
            loss[0].append(tran_loss)
            loss[1].append(rot_loss)
            # Output & Error
            pred_tran = pred['tran'].cpu().detach().numpy()
            pred_rot = pred['rot'].cpu().detach().numpy()
            gt_tran = gt_relative_pose['tran'].cpu().detach().numpy()
            gt_rot = gt_relative_pose['rot'].cpu().detach().numpy()
            if flags.loss_fn == 'R':
                pred_rot = pred_rot / np.linalg.norm(pred_rot, axis=1, keepdims=True)
            elif flags.loss_fn == 'C':
                pred_tran_sm = softmax(pred_tran, axis=1)
                assert(((np.abs(np.sum(pred_tran_sm, axis=1))-1)<1e-5).all())
                logits_sms['tran'].append(pred_tran_sm)
                cls_id = np.argmax(pred_tran_sm, axis=1)
                preds['tran_cls'].append(cls_id.reshape(-1,1))
                pred_tran = np.array(dataset.class2xyz(cls_id))

                pred_rot_sm = softmax(pred_rot, axis=1)
                assert(((np.abs(np.sum(pred_rot_sm, axis=1))-1)<1e-5).all())
                logits_sms['rot'].append(pred_rot_sm)
                cls_id = np.argmax(pred_rot_sm, axis=1)
                preds['rot_cls'].append(cls_id.reshape(-1,1))
                pred_rot = np.array(dataset.class2quat(cls_id))
            else:
                raise NotImplementedError
            gts['tran'].append(gt_tran)
            gts['rot'].append(gt_rot)
            if flags.loss_fn == 'C':
                gt_tran_cls = gt_tran_cls.cpu().detach().numpy()
                gts['tran_cls'].append(gt_tran_cls)
                gt_rot_cls = gt_rot_cls.cpu().detach().numpy()
                gts['rot_cls'].append(gt_rot_cls)

            preds['tran'].append(pred_tran)
            preds['rot'].append(pred_rot)

            d = np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1))
            np.putmask(d, d> 1, 1)
            np.putmask(d, d< -1, -1)
            # l2 distance between translation
            errors[0].append(np.linalg.norm(pred_tran-gt_tran, axis=1))
            # angle between rotation vectors
            errors[1].append(2 * np.arccos(d) * 180/math.pi)

        for key in preds.keys():
            if preds[key] == []:
                continue
            preds[key] = np.vstack(np.array(preds[key]))
        for key in gts.keys():
            if gts[key] == []:
                continue
            gts[key] = np.vstack(np.array(gts[key]))
        for key in logits_sms.keys():
            if logits_sms[key] == []:
                continue
            logits_sms[key] = np.vstack(np.array(logits_sms[key]))
            assert(((np.abs(np.sum(logits_sms[key], axis=1))-1)<1e-5).all())
        # evaluate loss
        loss = np.mean(np.array(loss), axis=1)
        # evaluate error
        errors = np.array([np.concatenate(errors[0]), np.concatenate(errors[1])])

        accuracy[0] = sum(_ < acc_threshold[0] for _ in errors[0])/len(errors[0])
        accuracy[1] = sum(_ < acc_threshold[1] for _ in errors[1])/len(errors[1])

        error_total = np.median(np.array(errors), axis=1)

        print('Loss: [tran, rot]', loss)
        print('Error [tran, rot]: ', error_total)
        print('Accuracy [tran, rot]:', accuracy)

        summary = {
            'loss': loss,
            'errors': errors,
            'preds': preds,
            'gts': gts,
            'logits_sms': logits_sms,
            'accuracy': accuracy,
        }
        return summary


def main():
    parser = argparse.ArgumentParser()
    # System settings
    parser.add_argument("--iteration", type=int, default=95000, help='evaluate model at iteration')
    parser.add_argument("--log_dir", type=str, default="./train_log", help='log dir')
    parser.add_argument("--exp_name", type=str, required=True, help='experiment name')
    parser.add_argument("--phase", type=str, default='test', help='phase to evaluate')
    args, _ = parser.parse_known_args()
    print(args)
    with open(os.path.join(args.log_dir, args.exp_name, 'README.txt'), 'r') as f:
        lines = f.read().splitlines()[1:]
        flags = json.loads(''.join(lines), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    flags = flags._replace(train_test_phase=args.phase)
    print(flags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_test, dataset = get_dataloader_benchmark(flags)

    model_path = os.path.join(flags.log_dir, flags.exp_name, 'model_{}.pth'.format(args.iteration))
    # Create save directory
    save_dir = os.path.join(flags.log_dir, flags.exp_name, 'model_{}_{}'.format(args.iteration, flags.train_test_phase))
    os.makedirs(save_dir, exist_ok=True)

    # gpus
    num_gpus = torch.cuda.device_count()
    print("[GPU number] {}".format(num_gpus))

    # Model
    model = CameraBranch(flags)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # Evaluate
    summary = run(model, dataloader_test, device, flags, dataset)
    f = open(os.path.join(save_dir, 'summary.pkl'), "wb")
    pickle.dump(summary, f)
    f.close()


if __name__ == '__main__':
    main()
