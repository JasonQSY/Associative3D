import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
from pyquaternion import Quaternion
from sklearn.externals import joblib
from tqdm import tqdm


def suncg_parse_path(dataset_dir, img_path):
    splits = img_path.split('/')
    house_id = splits[-2]
    img_id = splits[-1]
    img_path = os.path.join(dataset_dir, house_id, img_id)
    return img_path


class SUNCGRPDataset(data.Dataset):
    def __init__(self, flags):
        self.flags = flags
        if flags.loss_fn == 'C':
            assert(os.path.exists(flags.kmeans_trans_path))
            assert(os.path.exists(flags.kmeans_rots_path))
            self.kmeans_trans = joblib.load(flags.kmeans_trans_path)
            self.kmeans_rots = joblib.load(flags.kmeans_rots_path)
        anno_file = os.path.join(flags.split_dir, '{}_set.txt'.format(flags.train_test_phase))
        f = open(anno_file)
        lines = f.readlines()[3:]
        print("precomputing relative pose...")
        self.index = []
        compute_mean = True
        # get image size for network
        self.img_input_shape = tuple([int(_) for _ in flags.img_resize.split('x')])
        assert len(self.img_input_shape) == 2
        mean_npy = os.path.join(flags.cached_dir,
                "img_mean_{}x{}.npy".format(self.img_input_shape[0], self.img_input_shape[1]))
        if os.path.exists(mean_npy) or flags.train_test_phase != 'train':
            compute_mean = False
        all_img_resized = np.zeros((self.img_input_shape[1], self.img_input_shape[0], 3)) 
        print('Reading from '+flags.dataset_dir)
        for line in tqdm(lines):
            annot = line.split(' ')
            # get images and paths
            img1_path = suncg_parse_path(flags.dataset_dir, annot[0])
            img2_path = suncg_parse_path(flags.dataset_dir, annot[8])
            if compute_mean:
                img1, img2 = self.read_image(img1_path, img2_path)
                # calculate sum
                all_img_resized += img1
                all_img_resized += img2
            # calculate relative pose
            relative_pose = self.get_relative_pose(np.hstack((annot[1:8], annot[9:])).astype('f4'))
            relative_pose_dict = {'tran': relative_pose[:3], 'rot':relative_pose[3:]}
            self.index.append([img1_path, img2_path, relative_pose_dict])
        if compute_mean:
            # calculate mean
            all_img_resized /= (len(lines)*2)
            # save mean
            np.save(mean_npy, all_img_resized)
        else:
            all_img_resized = np.load(mean_npy)
        self.img_mean = all_img_resized
        assert(self.img_mean.shape == (self.img_input_shape[1], self.img_input_shape[0], 3))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.load_data(idx)

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

    def load_data(self, idx):
        img1_path, img2_path, relative_pose = self.index[idx]
        # preprocess images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.resize(img1, self.img_input_shape) - self.img_mean
        img2 = cv2.resize(img2, self.img_input_shape) - self.img_mean
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))
        # relative translation to class id
        x, y, z = relative_pose['tran']
        w, xi, yi, zi = relative_pose['rot']
        if self.flags.loss_fn == 'C':
            return {
                'img1': torch.FloatTensor(img1),
                'img2': torch.FloatTensor(img2),
                'relative_pose': {'tran': torch.FloatTensor(relative_pose['tran']),
                                  'rot': torch.FloatTensor(relative_pose['rot'])},
                'tran_cls': torch.cuda.LongTensor(self.xyz2class(x, y, z)),
                'rot_cls': torch.cuda.LongTensor(self.quat2class(w, xi, yi, zi))
            }
        else:
            return {
                'img1': torch.FloatTensor(img1),
                'img2': torch.FloatTensor(img2),
                'relative_pose': {'tran': torch.FloatTensor(relative_pose['tran']),
                                  'rot': torch.FloatTensor(relative_pose['rot'])},
                'tran_cls': torch.cuda.LongTensor([0]),
                'rot_cls': torch.cuda.LongTensor([0])
            }

    def get_relative_pose(self, pose):
        assert pose.shape[0] == 14
        q1 = Quaternion(pose[3:7])
        q2 = Quaternion(pose[10:14])
        t1 = pose[:3]
        t2 = pose[7:10]

        relative_rotation = (q2.inverse * q1).elements
        relative_translation = self.get_relative_T_in_cam2_ref(q2.inverse.rotation_matrix, t1, t2)
        rel_pose = np.hstack((relative_translation, relative_rotation))
        return rel_pose.reshape(-1)

    def get_relative_T_in_cam2_ref(self, R2, t1, t2):
        new_c2 = - np.dot(R2, t2)
        return np.dot(R2, t1) + new_c2

    def read_image(self, img1_path, img2_path):
        im1 = cv2.resize(cv2.imread(img1_path), self.img_input_shape)
        im2 = cv2.resize(cv2.imread(img2_path), self.img_input_shape)        
        return im1, im2

def get_dataloader(flags):
    dataset = SUNCGRPDataset(flags)
    return data.DataLoader(dataset, batch_size=flags.batch_size, 
                           shuffle=True, num_workers=flags.num_workers, 
                           pin_memory=False)

def get_dataloader_benchmark(flags):
    dataset = SUNCGRPDataset(flags)
    return data.DataLoader(dataset, batch_size=flags.batch_size, 
                           shuffle=False, num_workers=0, 
                           pin_memory=False), dataset

   

