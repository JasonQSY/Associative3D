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
from absl import app, flags
import cv2
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import random
import shutil

code_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(code_root)
sys.path.append(osp.join(code_root, '..'))
from relative3d.demo import siamese_demo_utils as demo_utils
from relative3d.renderer import utils as render_utils

torch.cuda.empty_cache()

flags.FLAGS(['demo'])
flags.DEFINE_string('output_dir', '/w/syqian/mesh_dir', 'output_dir')
#flags.DEFINE_string('name', 'exp7_siamese_bce64', 'experiment name')
#flags.DEFINE_string('num_train_epoch', 'latest', 'epoch name')

opts = flags.FLAGS
opts.batch_size = 1
#opts.num_train_epoch = 1
#opts.name = 'suncg_relnet_dwr_pos_ft'
opts.name = 'exp12_small'
opts.num_train_epoch = '8'
opts.classify_rot = True
opts.classify_dir = True
opts.pred_voxels = False# else:
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
split_file = '/home/syqian/p-voxels/script/v4_rpnet_split_relative3d_1rmPhouse/test_set.txt'
detection = True  # False if gt bbox
#detection = False
max_rois = 20


def clean_checkpoint_file(ckpt_file):
    checkpoint = torch.load(ckpt_file)
    keys = checkpoint.keys()
    
    temp = [key for key in keys if 'relative_quat_predictor' in key ] +  [key for key in keys if 'relative_encoder.encoder_joint_scale' in key]
    if len(temp) > 0:
        for t in temp:
            checkpoint.pop(t)

        torch.save(checkpoint, ckpt_file)


def prepare_inputs(house_id, view_name, img):
    # preprocess images
    img_fine = cv2.resize(img, (opts.img_width_fine, opts.img_height_fine))
    img_fine = np.transpose(img_fine, (2, 0, 1))
    img_coarse = cv2.resize(img, (opts.img_width, opts.img_height))
    img_coarse = np.transpose(img_coarse, (2, 0, 1))
    
    # read bbox
    gt_bbox_path = os.path.join(suncg_dir, 'bboxes_node', house_id, 
                                view_name + '_bboxes.mat')
    gt_bbox = sio.loadmat(gt_bbox_path)
    if detection:
        proposal_path = os.path.join(suncg_dir, 'edgebox_proposals', 
                                     house_id, view_name + '_proposals.mat')
        temp = sio.loadmat(proposal_path)
    else:
        temp = gt_bbox
        
    proposals = temp['proposals'][:, 0:4]
    gtInds = temp['gtInds']
        
    # prepare inputs
    inputs = {}
    inputs['img'] = torch.from_numpy(img_coarse/255.0).unsqueeze(0)
    inputs['img_fine'] = torch.from_numpy(img_fine/255.0).unsqueeze(0)
    if detection:
        inputs['bboxes'] = [torch.from_numpy(proposals)]
    inputs['empty'] = False
    #inputs['node_ids'] = gt_bbox['node_ids']
    
    return inputs

def forward_affinity(select_node_ids1, select_node_ids2, max_rois):
    """
    Calculate affinity matrix according to select_node_ids
    """
    affinity = np.zeros((max_rois, max_rois), dtype=np.float32)
    for i, node_id1 in enumerate(select_node_ids1):
        for j, node_id2 in enumerate(select_node_ids2):
            if node_id1 == node_id2:
                affinity[i, j] = 1

    return affinity

    
def main(_):
    # set random seed
    random.seed(2019)
    
    # load pre-trained model
    ckpt_file = '../cachedir/snapshots/{}/pred_net_{}.pth'.format(opts.name, opts.num_train_epoch)
    #clean_checkpoint_file(ckpt_file)
    tester = demo_utils.DemoTester(opts)
    tester.init_testing()
    
    print("Results saved in: {}".format(opts.output_dir))
    f = open(split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    #random.shuffle(lines)
    lines = lines[:100]
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
    
        # set up mesh_dir
        mesh_dir = os.path.join(opts.output_dir, house_id)
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        else:
            continue
        
        # view1
        inputs1 = prepare_inputs(house_id, view1, img1)
        tester.set_input(inputs1)
        objects = tester.predict_box3d()
        #bbox1 = tester.rois.cpu().data.numpy()
        #id1 = objects['id']
        id1 = tester.codes_pred_vis['id']
        bbox1 = tester.rois_pos_vis.cpu().numpy()
        roi_vis = render_utils.vis_detections(img1, bbox1[:, 1:])
        imageio.imwrite(os.path.join(mesh_dir, 'A_rois.png'), roi_vis)
        tester.save_codes_mesh(mesh_dir, tester.codes_pred_vis, 'A_code_pred')
        
        # view2
        inputs2 = prepare_inputs(house_id, view2, img2)
        tester.set_input(inputs2)
        objects = tester.predict_box3d()
        #bbox2 = tester.rois.cpu().data.numpy()
        #id2 = objects['id']
        id2 = tester.codes_pred_vis['id']
        bbox2 = tester.rois_pos_vis.cpu().numpy()
        roi_vis = render_utils.vis_detections(img2, bbox2[:, 1:])
        imageio.imwrite(os.path.join(mesh_dir, 'B_rois.png'), roi_vis)
        tester.save_codes_mesh(mesh_dir, tester.codes_pred_vis, 'B_code_pred')
        
        # affinity matrix
        valid_bs = 1
        affinity_pred = torch.zeros([max_rois, max_rois], dtype=torch.float32)
        affinity_pred = affinity_pred.cuda()
        objs_in_view1 = id1
        objs_in_view2 = id2
        norms = (torch.norm(objs_in_view1, p=2, dim=1) + 1e-3).detach()
        objs_in_view1 = objs_in_view1.div(norms.view(-1,1).expand_as(objs_in_view1))
        norms = (torch.norm(objs_in_view2, p=2, dim=1) + 1e-3).detach()
        objs_in_view2 = objs_in_view2.div(norms.view(-1,1).expand_as(objs_in_view2))
        prod = torch.mm(
            objs_in_view1, torch.t(objs_in_view2)
        ) * 5
        prod = torch.sigmoid(prod)
        sz_i = prod.shape[0]
        sz_j = prod.shape[1]
        if sz_i > 20 or sz_j > 20:
            shutil.rmtree(mesh_dir)
            continue
        affinity_pred[:sz_i, :sz_j] = prod
        
        match = affinity_pred.argmax(dim=1).cpu().numpy()
        affinity_pred = affinity_pred.cpu().detach().numpy()
        plt.figure()
        sns.heatmap(affinity_pred)
        plt.savefig(os.path.join(mesh_dir, 'affinity.png'))
        plt.close()


if __name__=='__main__':
    app.run(main)