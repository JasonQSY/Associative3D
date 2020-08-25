#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
import os.path as osp
import os
import scipy.misc
import scipy.io as sio
import torch
import imageio
import pdb
from absl import flags
import cv2
from tqdm import tqdm

code_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(code_root)
sys.path.append(osp.join(code_root, '..'))
from relative3d.demo import demo_utils

torch.cuda.empty_cache()

detection=True
flags.FLAGS(['demo'])
opts =  flags.FLAGS
opts.batch_size = 1
opts.num_train_epoch = 1
opts.name = 'suncg_relnet_dwr_pos_ft'
opts.classify_rot = True
opts.classify_dir = True
opts.pred_voxels = False# else:
#     inputs['bboxes'] = [torch.from_numpy(bboxes)]
#     inputs['scores'] = [torch.from_numpy(bboxes[:,0]*0+1)]

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

checkpoint = '../cachedir/snapshots/{}/pred_net_{}.pth'.format(opts.name, opts.num_train_epoch)
pretrained_dict = torch.load(checkpoint)
suncg_dir = '/w/syqian/suncg'
split_file = '/w/syqian/exp/rpnet/e11_rpnetfc_pytorch_debug/validation_set.txt'
output_dir = '/w/syqian/mesh_dir'


def clean_checkpoint_file(ckpt_file):
    checkpoint = torch.load(ckpt_file)
    keys = checkpoint.keys()
    
    temp = [key for key in keys if 'relative_quat_predictor' in key ] +  [key for key in keys if 'relative_encoder.encoder_joint_scale' in key]
    if len(temp) > 0:
        for t in temp:
            checkpoint.pop(t)

        torch.save(checkpoint, ckpt_file)

ckpt_file = '../cachedir/snapshots/{}/pred_net_{}.pth'.format(opts.name, opts.num_train_epoch)
clean_checkpoint_file(ckpt_file)

tester = demo_utils.DemoTester(opts)
tester.init_testing()

dataset = 'suncg'


def prepare_inputs(house_id, view_name, img):
    # preprocess images
    img_fine = cv2.resize(img, (opts.img_width_fine, opts.img_height_fine))
    img_fine = np.transpose(img_fine, (2, 0, 1))
    img_coarse = cv2.resize(img, (opts.img_width, opts.img_height))
    img_coarse = np.transpose(img_coarse, (2, 0, 1))
    
    # read proposal
    proposal_path = os.path.join(suncg_dir, 'edgebox_proposals', 
                                 house_id, view_name + '_proposals.mat')
    temp = sio.loadmat('./data/{}_proposals.mat'.format(dataset))
    proposals = temp['proposals'][:, 0:4]
    gtInds = temp['gtInds']
    
    # prepare inputs
    inputs = {}
    inputs['img'] = torch.from_numpy(img_coarse/255.0).unsqueeze(0)
    inputs['img_fine'] = torch.from_numpy(img_fine/255.0).unsqueeze(0)
    if detection:
        inputs['bboxes'] = [torch.from_numpy(proposals)]
    inputs['empty'] = False
    
    return inputs
    
    
def main():
    f = open(split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    #lines = lines[3:4]
    for line in tqdm(lines):
        # parse line
        splits = line.split(' ')
        img1_path = splits[0]
        img2_path = splits[8]
        house_id = img1_path.split('/')[-2]
        img1_name = img1_path.split('/')[-1]
        img2_name = img2_path.split('/')[-1]
        view1 = img1_name[:6]
        view2 = img2_name[:6]
        tqdm.write(str(house_id))
    
        # read image
        img1 = imageio.imread(img1_path)
        img2 = imageio.imread(img2_path)
    
        # inference
        mesh_dir = os.path.join(output_dir, house_id)
        
        inputs = prepare_inputs(house_id, view1, img1)
        tester.set_input(inputs)
        objects = tester.predict_box3d()
        tester.save_codes_mesh(mesh_dir, tester.codes_pred_vis, 'A_code_pred')
        
        inputs = prepare_inputs(house_id, view2, img2)
        tester.set_input(inputs)
        objects = tester.predict_box3d()
        tester.save_codes_mesh(mesh_dir, tester.codes_pred_vis, 'B_code_pred')

if __name__=='__main__':
    main()