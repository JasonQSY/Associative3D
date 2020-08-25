from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#code_root='/home/syqian/relative3d'
import sys
import numpy as np
import os.path as osp
import scipy.misc
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
code_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(code_root)
sys.path.append(osp.join(code_root, '..'))
import pdb
from absl import flags
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

img = scipy.misc.imread('./data/{}_img.png'.format(dataset))

img_fine = scipy.misc.imresize(img, (opts.img_height_fine, opts.img_width_fine))
img_fine = np.transpose(img_fine, (2,0,1))

img_coarse = scipy.misc.imresize(img, (opts.img_height, opts.img_width))
img_coarse = np.transpose(img_coarse, (2,0,1))


temp = sio.loadmat('./data/{}_proposals.mat'.format(dataset))

proposals = temp['proposals'][:, 0:4]
gtInds = temp['gtInds']
# bboxes = sio.loadmat('./data/{}_bboxes_1.mat'.format(dataset))['bboxes'].astype(np.float)
inputs = {}
inputs['img'] = torch.from_numpy(img_coarse/255.0).unsqueeze(0)
inputs['img_fine'] = torch.from_numpy(img_fine/255.0).unsqueeze(0)
if detection:
    inputs['bboxes'] = [torch.from_numpy(proposals)]

inputs['empty'] = False
tester.set_input(inputs)


objects = tester.predict_box3d()
visuals = tester.render_outputs()


f, axarr = plt.subplots(2, 2, figsize=(20, 8))
axarr[0, 0].imshow(visuals['img'])
axarr[0, 0].axis('off')
axarr[0, 1].imshow(visuals['b_pred_objects_cam_view'])
axarr[0, 1].axis('off')
axarr[1, 0].imshow(visuals['img_roi'])
axarr[1, 0].axis('off')
axarr[1, 1].imshow(visuals['b_pred_scene_cam_view'])
axarr[1, 1].axis('off')
plt.show()
plt.savefig('hello.png')
