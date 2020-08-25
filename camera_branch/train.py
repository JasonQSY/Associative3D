import argparse
import git
import json
import math
import numpy as np
import os
import time
import torch

from copy import deepcopy
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from network import CameraBranch
from suncg import get_dataloader, get_dataloader_benchmark

torch.multiprocessing.set_start_method('spawn',force=True)

def evaluate(model, dataloader, device, dataset):
    loss = [[], []]
    error = [[], []]
    accuracy = [[], []]
    acc_threshold = [1.0, 30.0] # threshold for translation and rotation error to say prediction is correct.
    model = model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img1 = batch['img1']
            img2 = batch['img2']
            img1 = img1.to(device)
            img2 = img2.to(device)
            gt_relative_pose = batch['relative_pose']
            gt_relative_pose['tran'] = gt_relative_pose['tran'].to(device)
            gt_relative_pose['rot'] = gt_relative_pose['rot'].to(device)
            gt_tran_cls = batch['tran_cls']
            gt_rot_cls = batch['rot_cls']
            gt_tran_cls = gt_tran_cls.to(device)
            gt_rot_cls = gt_rot_cls.to(device)
            pred = model(img1, img2)
            if dataset.flags.loss_fn == 'R':
                _, tran_loss, rot_loss = model.module.regression_loss(pred, gt_relative_pose)
            elif dataset.flags.loss_fn == 'C':
                _, tran_loss, rot_loss = model.module.classification_loss(pred, gt_tran_cls, gt_rot_cls)
            else:
                raise NotImplementedError
            tran_loss = tran_loss.item()
            rot_loss = rot_loss.item()
            loss[0].append(tran_loss)
            loss[1].append(rot_loss)
        
            pred_tran = pred['tran'].cpu().detach().numpy()
            pred_rot = pred['rot'].cpu().detach().numpy()
            gt_relative_pose['tran'] = gt_relative_pose['tran'].cpu().detach().numpy()
            gt_relative_pose['rot'] = gt_relative_pose['rot'].cpu().detach().numpy()
            if dataset.flags.loss_fn == 'R':
                pred_rot = pred_rot / np.linalg.norm(pred_rot, axis=1, keepdims=True)
            elif dataset.flags.loss_fn == 'C':
                pred_tran = np.argmax(pred_tran, axis=1)
                pred_tran = dataset.class2xyz(pred_tran)
                pred_tran = np.array(pred_tran)
                pred_rot = np.argmax(pred_rot, axis=1)
                pred_rot = dataset.class2quat(pred_rot)
                pred_rot = np.array(pred_rot)
            else:
                raise NotImplementedError

            # l2 distance between translation
            error[0].append(np.linalg.norm(pred_tran - gt_relative_pose['tran'], axis=1))
            # angle between rotation vectors
            d = np.abs(np.sum(np.multiply(pred_rot, gt_relative_pose['rot']), axis=1))
            np.putmask(d, d > 1, 1)
            np.putmask(d, d < -1, -1)
            error[1].append(2 * np.arccos(d) * 180/math.pi)
    # validation loss
    loss = np.mean(np.array(loss), axis=1)
    # validation error
    error = [np.concatenate(error[0]), np.concatenate(error[1])]

    accuracy[0] = sum(_ < acc_threshold[0] for _ in error[0])/len(error[0])
    accuracy[1] = sum(_ < acc_threshold[1] for _ in error[1])/len(error[1])

    error = np.median(np.array(error), axis=1)

    print('Loss: [tran, rot]', loss)
    print('Error [tran, rot]: ', error)
    print('Accuracy [tran, rot]:', accuracy)
    model = model.train()
    return loss, error, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda vll: v.lower() == "true")

    # Training parameters
    parser.add_argument("--img_resize", type=str, default = "224x224",
                        help='image preprocessing, options: 455x256, 224x224')
    parser.add_argument("--base_network", type=str, default="resnet50",
                        help='base network for siamese network, options: GoogleNet, resnet18, resnet50')
    parser.add_argument("--loss_fn", type=str, default="C",
                        help='options: R: regression, C: classification')
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Initial learning rate")
    parser.add_argument("--optimization", default= 'sgdm', type=str, help= "use adadelta, sgd, sgdm")
    parser.add_argument("--lr_scheduler", type = str, default='StepLR', help = 'torch.optim.lr_scheduler')
    parser.add_argument("--lr_decay_rate", type = float, default = 0.90, help = 'decay rate for learning rate')
    parser.add_argument("--lr_decay_step", type = int, default = 500, help = 'epoch to decay learning rate, use big number of step to avoid decaying the lr')
    parser.add_argument("--nb_epoches", type = int, default = 1000, help = "Nb of epoches to train the model, alternative to --max-step")
    parser.add_argument("--batch_size", type = int, default = 32, help="Number of example per batch")
    parser.add_argument("--wd", type = float, default = 1e-2, help="weight decay on each layer, default = 0.0 mean no decay")
    parser.add_argument("--beta", type = int, default = 1, help="beta to weight between the two losses as in the paper")

    # Data and log directory
    parser.add_argument("--dataset_dir", type=str, default="./suncg_dataset", help="dataset directory")
    parser.add_argument("--split_dir", type=str, default="./split", help="dataset split")
    parser.add_argument("--log_dir", type=str, default="./train_log", help='log dir')
    parser.add_argument("--exp_name", type=str, required=True, help='experiment name')
    parser.add_argument("--dataset", default = 'suncg', type=str, help ='dataset name to train the network')
    parser.add_argument("--cached_dir", default = './cached_dir', type=str, help ='folder to save cached files')

    # Tensorboard logging
    parser.add_argument("--train_test_phase", default = 'train', help = 'train, validation or test')
    parser.add_argument("--save_freq", type=int, default=5000, help='number of iterations model is saved')
    parser.add_argument("--val_freq", type=int, default=5000, help='number of iterations model is evaluated')
    parser.add_argument("--trainlog_freq", type=int, default=100, help='number of iterations train loss is saved')

    # System settings
    parser.add_argument("--continue_training", type=int, default=0,
                        help='continue training on the logdir')
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers in dataloader')

    # Classification metric
    parser.add_argument("--kmeans_trans_path", type=str, default='./cached_dir/kmeans_trans_60.pkl', help='xyz2cls model')
    parser.add_argument("--kmeans_rots_path", type=str, default='./cached_dir/kmeans_rots_30.pkl', help='quat2cls model')

    flags, _ = parser.parse_known_args()
    print(flags)
    print(f"Log into {os.path.join(flags.log_dir, flags.exp_name)}")
    os.makedirs(os.path.join(flags.log_dir, flags.exp_name), exist_ok=True)
    with open(os.path.join(flags.log_dir, flags.exp_name, 'README.txt'), 'w') as f:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        f.write("git hexsha: "+sha+'\n')
        json.dump(flags.__dict__, f, indent=2)

    writer = SummaryWriter(os.path.join(flags.log_dir, flags.exp_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader(flags)
    val_flags = deepcopy(flags)
    val_flags.train_test_phase = 'validation'
    dataloader_val, dataset = get_dataloader_benchmark(val_flags)

    # gpus
    num_gpus = torch.cuda.device_count()
    print("[GPU number] {}".format(num_gpus))

    # Model
    model = CameraBranch(flags)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=flags.lr, momentum=0.9, nesterov = True)

    num_iter = 0

    if flags.continue_training != 0:
        num_iter = flags.continue_training
        model_path = os.path.join(flags.log_dir, flags.exp_name, 'model_{}.pth'.format(flags.continue_training))
        model.load_state_dict(torch.load(model_path))

    # LR Scheduler
    iteration_per_epoch = len(dataloader) // flags.batch_size
    if flags.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, iteration_per_epoch*flags.lr_decay_step,
                                        gamma=flags.lr_decay_rate, last_epoch=-1)
        for _ in range(num_iter):
            scheduler.step()
    else:
        raise NotImplementedError

    for epoch_id in range(flags.nb_epoches):
        print("iter {}".format(num_iter))
        for batch in tqdm(dataloader):
            img1 = batch['img1']
            img2 = batch['img2']
            img1 = img1.to(device)
            img2 = img2.to(device)
            gt_relative_pose = batch['relative_pose']
            gt_tran_cls = batch['tran_cls']
            gt_rot_cls = batch['rot_cls']
            gt_relative_pose['tran'] = gt_relative_pose['tran'].to(device)
            gt_relative_pose['rot'] = gt_relative_pose['rot'].to(device)
            gt_tran_cls = gt_tran_cls.to(device)
            gt_rot_cls = gt_rot_cls.to(device)

            # start optimizer
            optimizer.zero_grad()
            pred = model(img1, img2)
            if flags.loss_fn == 'R':
                total_loss, tran_loss, rot_loss = model.module.regression_loss(pred, gt_relative_pose)
            elif flags.loss_fn == 'C':
                total_loss, tran_loss, rot_loss = model.module.classification_loss(pred, gt_tran_cls, gt_rot_cls)
            else:
                raise NotImplementedError
            if num_iter % flags.trainlog_freq == 0:
                # log to tensorboard
                writer.add_scalar('train/tran_Loss', tran_loss.item(), num_iter)
                writer.add_scalar('train/rot_loss', rot_loss.item(), num_iter)
                writer.add_scalar('learning_rate', np.array(scheduler.get_lr()), num_iter)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            num_iter += 1
            if num_iter % flags.save_freq == 0:
                print('Saving model at iter {}...'.format(num_iter))
                start = time.time()
                model_path = os.path.join(flags.log_dir, flags.exp_name, 'model_{}.pth'.format(num_iter))
                torch.save(model.state_dict(), model_path)
                end = time.time()
                print('save_model time elapsed: {}s'.format(end-start))
            if num_iter % flags.val_freq == 0:
                print('Start validation at iter {}...'.format(num_iter))
                start = time.time()
                val_loss, val_error, val_acc = evaluate(model, dataloader_val, device, dataset)
                writer.add_scalar('Validation/Loss_tran', val_loss[0], num_iter)
                writer.add_scalar('Validation/Loss_rot', val_loss[1], num_iter)
                writer.add_scalar('Validation/Error_tran', val_error[0], num_iter)
                writer.add_scalar('Validation/Error_rot', val_error[1], num_iter)
                writer.add_scalar('Validation/Acc_tran', val_acc[0], num_iter)
                writer.add_scalar('Validation/Acc_rot', val_acc[1], num_iter)
                end = time.time()
                print('Validation time elapsed: {}s'.format(end-start))


if __name__=='__main__':
    main()
