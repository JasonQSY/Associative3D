import argparse
import cv2
import json
import numpy as np
import os
import pickle
import torch

from argparse import Namespace
from scipy.special import softmax
from sklearn.externals import joblib
from pyquaternion import Quaternion
from tqdm import tqdm

from network import CameraBranch


class Camera_Branch_Inference():
    def __init__(self, cfg, device):
        self.cfg = cfg
        # img preprocess
        self.img_input_shape = tuple([int(_) for _ in cfg.img_resize.split('x')])
        self.img_mean = np.load(cfg.img_mean)
        # device
        self.device = device
        # Model
        self.model = CameraBranch(cfg)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(cfg.model_weight))
        self.model = self.model.eval()
        self.model = self.model.to(device)
        # bin -> vectors
        self.kmeans_trans = joblib.load(cfg.kmeans_trans_path)
        self.kmeans_rots = joblib.load(cfg.kmeans_rots_path)

    def inference(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.resize(img1, self.img_input_shape) - self.img_mean
        img2 = cv2.resize(img2, self.img_input_shape) - self.img_mean
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))
        img1 = torch.FloatTensor([img1]).to(self.device)
        img2 = torch.FloatTensor([img2]).to(self.device)
        with torch.no_grad():
            pred = self.model(img1, img2)
        pred_tran = pred['tran'].cpu().detach().numpy()
        pred_rot = pred['rot'].cpu().detach().numpy()
        pred_tran_sm = softmax(pred_tran, axis=1)
        pred_rot_sm = softmax(pred_rot, axis=1)
        pred_sm = {'rot': pred_rot_sm, 'tran': pred_tran_sm}
        return pred_sm

    def xyz2class(self, x, y, z):
        return self.kmeans_trans.predict([[x,y,z]])

    def quat2class(self, w, xi, yi, zi):
        return self.kmeans_rots.predict([[w, xi, yi, zi]])

    def class2xyz(self, cls):
        assert((cls >= 0).all() and (cls < self.kmeans_trans.n_clusters).all())
        return self.kmeans_trans.cluster_centers_[cls]

    def class2quat(self, cls):
        assert((cls >= 0).all() and (cls < self.kmeans_rots.n_clusters).all())
        return self.kmeans_rots.cluster_centers_[cls]


def get_relative_pose(pose):
    assert pose.shape[0] == 14
    q1 = Quaternion(pose[3:7])
    q2 = Quaternion(pose[10:14])
    t1 = pose[:3]
    t2 = pose[7:10]

    relative_rotation = (q2.inverse * q1).elements
    relative_translation = get_relative_T_in_cam2_ref(q2.inverse.rotation_matrix, t1, t2)
    rel_pose = np.hstack((relative_translation, relative_rotation))
    return rel_pose.reshape(-1)


def get_relative_T_in_cam2_ref(R2, t1, t2):
    new_c2 = - np.dot(R2, t2)
    return np.dot(R2, t1) + new_c2


def suncg_parse_path(dataset_dir, img_path):
    splits = img_path.split('/')
    house_id = splits[-2]
    img_id = splits[-1]
    img_path = os.path.join(dataset_dir, house_id, img_id)
    return img_path


def inference_by_dataset(cam_model, split_file, log_dir, dataset_dir):
    summary = {}
    with open(split_file, 'r') as f:
        lines = f.readlines()[3:]
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'summary.pkl')

    for line in tqdm(lines):
        annot = line.split(' ')
        img1_path, img2_path = annot[0], annot[8]
        house_idx = img1_path.split('/')[-2]
        cam1_idx = img1_path.split('/')[-1].split('_')[0]
        cam2_idx = img2_path.split('/')[-1].split('_')[0]
        key = house_idx + '_' + cam1_idx + '_' + cam2_idx
        gt_relative_pose = get_relative_pose(np.hstack((annot[1:8], annot[9:])).astype('f4'))
        prediction = cam_model.inference(suncg_parse_path(dataset_dir, img1_path), 
                                     suncg_parse_path(dataset_dir, img2_path))
        summary[key] = {
            'tran_gt': gt_relative_pose[:3], 
            'rot_gt':  gt_relative_pose[3:],
            'tran_logits': prediction['tran'],
            'rot_logits': prediction['rot'],
            'tran_pred': cam_model.class2xyz(np.argmax(prediction['tran'])),
            'rot_pred': cam_model.class2quat(np.argmax(prediction['rot'])),
        }
    with open(log_file_path, 'wb') as log_f:
        pickle.dump(summary, log_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1-path", type=str, help='path to img 1', default='./example/000009_mlt.png')
    parser.add_argument("--img2-path", type=str, help='path to img 2', default='./example/000029_mlt.png')
    parser.add_argument("--config-path", type=str, default='./config.txt', help='path to config')
    parser.add_argument("--log-dir", type=str, default="./output", help='log dir')
    parser.add_argument("--split-file", type=str, default="", help='split file path')
    parser.add_argument("--dataset-dir", type=str, default="./suncg_dataset", help="dataset directory")
    args, _ = parser.parse_known_args()
    print(args)
    with open(args.config_path, 'r') as f:
        cfg = Namespace(**json.load(f))
    print(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam_model = Camera_Branch_Inference(cfg, device)
    if len(args.split_file) == 0:
        result = cam_model.inference(args.img1_path, args.img2_path)
        print(f"Predicted top 1 translation: {cam_model.class2xyz(np.argmax(result['tran']))}")
        print(f"Predicted top 1 rotation: {cam_model.class2quat(np.argmax(result['rot']))}")
    else:
        inference_by_dataset(cam_model, args.split_file, args.log_dir, args.dataset_dir)
    

if __name__ == '__main__':
    main()
