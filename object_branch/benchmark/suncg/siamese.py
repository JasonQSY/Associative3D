from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import time
import scipy.misc
import pdb
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.io as sio
import imageio
import itertools
from tqdm import tqdm

from ...data import suncg_multi as suncg_data
from ...utils import suncg_parse
from ...nnutils import train_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import oc_net
from ...nnutils import disp_net
from ...utils import visutil
from ...renderer import utils as render_utils
from ...utils import quatUtils


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')

flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')

flags.DEFINE_boolean('pred_id', True, 'predict object id embedding?')
flags.DEFINE_integer('nz_id', 40, 'the dimension of object id embedding')
flags.DEFINE_string('id_loss_type', 'bce', 'bce or mse')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')
flags.DEFINE_boolean('shape_dec_ft', False, 'If predicting voxels, should we pretrain from an existing deocder')

flags.DEFINE_string('no_graph_pretrain_name', 'box3d_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('no_graph_pretrain_epoch', 0, 'Network epoch from which we will finetune')

flags.DEFINE_string('ft_pretrain_name', 'box3d_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('ft_pretrain_epoch', 0, 'Network epoch from which we will finetune')
flags.DEFINE_string('set', 'val', 'dataset to use')
flags.DEFINE_integer('max_rois', 10, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 40, 'If we have more objects than this per batch, we will reject the batch.')
flags.DEFINE_boolean('use_class_weights', False, 'Use class weights')
flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')

FLAGS = flags.FLAGS


def affinity_eval(id_pred, im2obj, affinity_gt, max_rois, loss_type='bce', class_pred=None):
    """
    id_pred: (num_object x nz_id)
    im2obj: num_objects for each image, len(im2obj) = batch_size * 2
    affinity_gt: provided by dataloader
    max_rois: opts.max_rois
    loss_type: 'mse' or 'bce' 
    """
    topk_match = 5
    thres = 0.75
    #thres = 0.5

    total = 0
    corr = 0
    im2obj = np.array(im2obj)
    id_upper = np.array(list(itertools.accumulate(im2obj)))
    id_lower = id_upper - im2obj
    valid_bs = affinity_gt.shape[0]
    #affinity_pred = torch.zeros([valid_bs, max_rois, max_rois], dtype=torch.float32)
    #affinity_pred = affinity_pred.cuda()
    for i in range(valid_bs):
        objs_in_view1 = id_pred[id_lower[i]:id_upper[i]]
        norms = (torch.norm(objs_in_view1, p=2, dim=1) + 1e-3).detach()
        objs_in_view1 = objs_in_view1.div(norms.view(-1,1).expand_as(objs_in_view1))
        objs_in_view2 = id_pred[id_lower[i + valid_bs]:id_upper[i + valid_bs]]
        norms = (torch.norm(objs_in_view2, p=2, dim=1) + 1e-3).detach()
        objs_in_view2 = objs_in_view2.div(norms.view(-1,1).expand_as(objs_in_view2))
        prod = torch.mm(
            objs_in_view1, torch.t(objs_in_view2)
        ) * 5.0
        prod = torch.sigmoid(prod)
        #prod /= 2
        #prod += 0.5
        #prod = torch.clamp(prod, min=0, max=1)
        sz_i = prod.shape[0]
        sz_j = prod.shape[1]
        
        # gt
        gt_m = affinity_gt[i, :sz_i, :sz_j].cpu().numpy()
        no_matching = np.max(gt_m, axis=1) <= thres
        gt = np.argmax(gt_m, axis=1)
        gt[no_matching] = -1
        
        # prediction
        pred_m = prod.cpu().detach().numpy()
        no_matching = np.max(pred_m, axis=1) <= thres
        
        # top-1
        """
        pred = np.argmax(pred_m, axis=1)
        pred[no_matching] = -1
        this_total = (pred == gt).size
        this_corr = (pred == gt).sum()
        corr += this_corr
        total += this_total
        """

        # top-K
        pred = np.argsort(pred_m, axis=1)[:, ::-1][:, :topk_match]
        #real_k = min(pred.shape[1], topk_match)
        for i in range(pred.shape[0]):
            this_corr = False
            for j in range(pred.shape[1]):
                if pred_m[i, pred[i][j]] < thres:
                    continue
                if pred[i][j] == gt[i]:
                    this_corr = True
            total += 1
            if this_corr:
                corr += 1
            if no_matching[i] and gt[i] == -1:
                corr += 1

        #pred[no_matching, :] = -1
        #this_total = (pred[:, 0] == gt).size
        #this_corr = pred[:, 0] == gt
        #for k in range(real_k):
        #    this_corr = np.logical_or(this_corr, pred[:, k] == gt)
        #this_corr = this_corr.sum()
        #corr += this_corr
        #total += this_total

    return corr, total


class SiameseTester(train_utils.Trainer):
    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        opts = self.opts
        assert (not (opts.ft_pretrain_epoch > 0 and opts.num_pretrain_epochs > 0))

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)

        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width), opts=self.opts,
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=opts.pred_voxels, nz_shape=opts.nz_shape,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot,
            pred_id=opts.pred_id, nz_id=opts.nz_id,
            b_size=opts.batch_size * 2)

        if opts.ft_pretrain_epoch > 0:
            print("Loading previous model for fine tune")
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.ft_pretrain_name)
            self.load_network(
                self.model, 'pred', opts.ft_pretrain_epoch, network_dir=network_dir, strict=False)
            
        if opts.pred_voxels:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        if opts.pred_voxels and opts.shape_dec_ft:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.model.code_predictor.shape_predictor.decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)

        if self.opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs - 1, strict=False)
            
            
        # freeze all layers except IDPredictor
        #print("freezing layers except IDPredictor")
        #for param in self.model.parameters():
        #    param.requires_grad = False
        #for param in self.model.code_predictor.id_predictor.parameters():
        #    param.requires_grad = True
        #print("embedding dim {}".format(opts.nz_id))
        #print("object dim {}".format(opts.nz_feat))
        
        self.model = self.model.cuda()
        self.lsopt = loss_utils.LeastSquareOpt()
        self.stored_relative_quats  = []
        self.stored_translation_dependent_relative_angle  = []
        self.loss_per_batch = []
        self.corr = 0
        self.total = 0
        return

    def init_dataset(self):
        opts = self.opts
        self.real_iter = 1  # number of iterations we actually updated the net for
        self.data_iter = 1  # number of iterations we called the data loader
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        if opts.set == 'val':
            self.train_mode = False
            self.model.eval()
        else:
            rng = np.random.RandomState(10)
            rng.shuffle(self.split[opts.set])
            len_splitset = int(len(self.split[opts.set])*opts.split_size)
            self.split[opts.set] = self.split[opts.set][0:len_splitset]

        #self.dataloader = suncg_data.suncg_data_loader(self.split[opts.set], opts)
        self.dataloader = suncg_data.suncg_data_loader_benchmark(self.split[opts.set], opts)
       
        if not opts.pred_voxels:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.voxel_encoder,
                'encoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.load_network(
                self.voxel_decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.voxel_encoder.eval()
            self.voxel_encoder = self.voxel_encoder.cuda()
            self.voxel_decoder.eval()
            self.voxel_decoder = self.voxel_decoder.cuda()

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64 // opts.voxel_size, use_max=True, batch_mode=True
            ).cuda()

        self.spatial_image = Variable(suncg_data.define_spatial_image(opts.img_height_fine, opts.img_width_fine, 1.0/16).unsqueeze(0).cuda()) ## (1, 2, 30, 40)
        
        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = Variable(self.quat_medoids).cuda()

        if opts.classify_dir:
            self.direction_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
            self.direction_medoids_var = Variable(self.direction_medoids).cuda()

    def define_criterion(self):
        self.smoothed_factor_losses = {
            'shape': 0,
            'scale': 0,
            'quat': 0,
            'trans': 0,
            'rel_trans': 0,
            'rel_scale': 0,
            'rel_quat': 0,
            'class': 0,
            'var_mean': 0,
            'var_std': 0,
            'var_mean_rel_dir': 0,
            'var_std_rel_dir': 0,
            'affinity': 0
        }
        
    def set_input(self, batch):
        opts = self.opts

        if batch is None or not batch:
            self.invalid_batch = True
            return
        
        # siamese part
        if 'gt_affinity' in batch:
            self.affinity_gt = batch['gt_affinity'].cuda()
        else:
            self.invalid_batch = True
            return
        
        # relnet part
        # old relative3d code
        batch = batch['views']
        rois = suncg_parse.bboxes_to_rois(batch['bboxes'])
        self.data_iter += 1

        if rois.numel() <= 5 or rois.numel() > (5 * opts.max_total_rois) * (
                2.0 * opts.batch_size) / 8.0:  # with just one element, batch_norm will screw up
            self.invalid_batch = True
            return
        else:
            pairs = self.count_number_pairs(rois)
            if self.opts.only_pairs and pairs <= 1:
                self.invalid_batch = True
                return
            self.invalid_batch = False
            self.real_iter += 1

        # extra_codes = batch['extra_codes']
        # self.amodal_bboxes = torch.stack(extra_codes[0][5]).type(torch.FloatTensor).round()
        self.house_names = batch['house_name']
        self.view_ids = batch['view_id']
        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)
        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(), requires_grad=False)

        # object_classes = []
        # cam2voxs = []
        # for exc in extra_codes:
        #     for obj_classes in exc[4]:
        #         object_classes.append(obj_classes)
        #     for cam2vox in exc[6]:
        #         cam2voxs.append(cam2vox)
        
        # object_classes = torch.stack(object_classes)
        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors_quats = code_tensors['quat']
        object_classes = code_tensors['class'].type(torch.LongTensor)
        self.class_gt = self.object_classes = Variable(object_classes.cuda(), requires_grad=False)
        # code_tensors, code_tensors_quats = suncg_parse.collate_codes(batch['codes'])  ## Won't output quats
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1)  # unsqueeze voxels

        common_src_classes = []
        common_trj_classes = []
        self.common_masks = []
        for bx, ocs in enumerate(suncg_parse.batchify(object_classes, self.rois[:, 0].data.cpu())):
            nx = len(ocs)
            for ix in range(nx):
                for jx in range(nx):
                    common_src_classes.append(ocs[ix])
                    common_trj_classes.append(ocs[jx])

        common_src_classes = Variable(torch.cat(common_src_classes).cuda(), requires_grad=False)
        common_trj_classes = Variable(torch.cat(common_trj_classes).cuda(), requires_grad=False)
        self.common_classes = [common_src_classes, common_trj_classes]
        self.object_locations = suncg_parse.batchify(code_tensors['trans'], self.rois[:, 0].data.cpu())

        cam2voxs = code_tensors['transform_cam2vox']
        cam2voxs = suncg_parse.batchify(cam2voxs, self.rois[:,0].data.cpu())
        relative_direction_rotation = []
        relative_dir_mask = []
        for bx in range(len(cam2voxs)):
            nx = len(cam2voxs[bx])
            mask = torch.ones([nx*nx])
            for ix in range(nx):
                for jx in range(nx):
                    if ix == jx:
                        mask[ix*nx + jx] = 0
                    directions = []
                    for cam2vox in cam2voxs[bx][ix]:
                        dt_trj_cam_frame = self.object_locations[bx][jx]
                        direction = self.homogenize_coordinates(dt_trj_cam_frame.unsqueeze(0))
                        direction = torch.matmul(cam2vox, direction.t()).t()[:,0:3]
                        direction = direction/(torch.norm(direction,p=2, dim=1, keepdim=True) + 1E-5)
                        directions.append(direction)
                    directions = torch.cat(directions)
                    relative_direction_rotation.append(directions)
            relative_dir_mask.append(mask)
        self.relative_dir_mask = Variable(torch.cat(relative_dir_mask,dim=0).byte()).cuda()

        if opts.classify_dir and not opts.gmm_dir:
            relative_direction_rotation_binned  = [suncg_parse.directions_to_bininds(t, self.direction_medoids) for t in relative_direction_rotation]
            relative_direction_rotation_directions = [suncg_parse.bininds_to_directions(t, self.direction_medoids) for t in relative_direction_rotation_binned]
            self.relative_direction_rotation_unquantized = [Variable(t).cuda() for t in relative_direction_rotation]
            self.relative_direction_rotation = [Variable(t).cuda() for t in relative_direction_rotation_binned]
        else:
            self.relative_direction_rotation = [Variable(t).cuda() for t in relative_direction_rotation]

        if opts.classify_rot and not opts.gmm_rot:
            code_tensors_quats = [suncg_parse.quats_to_bininds(code_quat, self.quat_medoids) for code_quat in code_tensors_quats]

        self.codes_gt_quats = [
            Variable(t.cuda(), requires_grad=False) for t in code_tensors_quats]
        codes_gt_keys = ['shape', 'scale', 'trans']
        self.codes_gt  ={key : Variable(code_tensors[key].cuda(), requires_grad=False) 
                            for key in codes_gt_keys}
        self.codes_gt['quat'] = self.codes_gt_quats
        
        self.object_locations = [Variable(t_.cuda(), requires_grad=False) for t_ in
                                 self.object_locations]
        self.object_scales = suncg_parse.batchify(code_tensors['scale'] + 1E-10, self.rois[:, 0].data.cpu())
        self.object_scales = [Variable(t_.cuda(), requires_grad=False) for t_ in
                                 self.object_scales]


        self.relative_trans_gt = []
        self.relative_scale_gt = []
        for bx in range(len(self.object_locations)):
            relative_locations = self.object_locations[bx].unsqueeze(0) - self.object_locations[bx].unsqueeze(1)
            relative_locations = relative_locations.view(-1, 3)
            self.relative_trans_gt.append(relative_locations)
            relative_scales = self.object_scales[bx].unsqueeze(0).log() - self.object_scales[bx].unsqueeze(1).log() ## this is in log scale.
            relative_scales = relative_scales.view(-1, 3)
            self.relative_scale_gt.append(relative_scales)

        self.relative_scale_gt = torch.cat(self.relative_scale_gt, dim=0) # this is in log scale
        self.relative_trans_gt = torch.cat(self.relative_trans_gt, dim=0)
        if self.downsample_voxels:
            self.codes_gt['shape'] = self.downsampler.forward(self.codes_gt['shape'])

        if not opts.pred_voxels:
            self.codes_gt['shape'] = self.voxel_encoder.forward(self.codes_gt['shape'])
        
        self.relative_gt = {'relative_trans' : self.relative_trans_gt,
                            'relative_scale' : self.relative_scale_gt,
                            'relative_dir' : self.relative_direction_rotation,
                            'relative_mask' : self.relative_dir_mask,
                            }
        
        # use object locations to split batch
        self.im2obj = list([i.shape[0] for i in self.object_locations])
        
        return


    def get_current_scalars(self):
        loss_dict = {'total_loss': self.smoothed_total_loss, 'iter_frac': self.real_iter / self.data_iter}
        for k in self.smoothed_factor_losses.keys():
            # if np.abs(self.smoothed_factor_losses[k]) > 1E-5 or 'loss_{}'.format(k) in loss_dict.keys():
            loss_dict['loss_' + k] = self.smoothed_factor_losses[k]
        return loss_dict

    def render_codes(self, code_vars, prefix='mesh'):
        opts = self.opts
        code_list = suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), self.rois.data.cpu()[:, 0])
        mesh_dir = osp.join(opts.rendering_dir, opts.name)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        new_codes_list = suncg_parse.convert_codes_list_to_old_format(code_list[0])
        render_utils.save_parse(mesh_file, new_codes_list, save_objectwise=False)
        png_dir = mesh_file.replace('.obj', '/')
        render_utils.render_mesh(mesh_file, png_dir)

        return imageio.imread(osp.join(png_dir, prefix + '_render_000.png'))

    def get_current_visuals(self):
        visuals = {}
        opts = self.opts
        visuals['img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        codes_gt_vis = {k:t for k,t in self.codes_gt.items()}
        if not opts.pred_voxels:
            codes_gt_vis['shape'] = torch.sigmoid(
                self.voxel_decoder.forward(self.codes_gt['shape'])
            )

        if opts.classify_rot:
            codes_gt_vis['quat'] = [Variable(suncg_parse.bininds_to_quats(d.cpu().data, self.quat_medoids), requires_grad=False) for d in codes_gt_vis['quat']]
        
        visuals['codes_gt'] = self.render_codes(codes_gt_vis, prefix='gt')

        codes_pred_vis = {k:t for k,t in self.codes_pred.items()}
        if not opts.pred_voxels:
            codes_pred_vis['shape'] = torch.sigmoid(
                self.voxel_decoder.forward(self.codes_pred['shape'])
            )

        if opts.classify_rot:
            _, bin_inds = torch.max(codes_pred_vis['quat'].data.cpu(), 1)
            codes_pred_vis['quat'] = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)
        visuals['codes_pred'] = self.render_codes(codes_pred_vis, prefix='pred')

        return visuals

    def get_current_points(self):
        pts_dict = {}
        return pts_dict

    def forward(self):
        opts = self.opts
        feed_dict = {}
        feed_dict['imgs_inp_fine'] = self.input_imgs_fine
        feed_dict['imgs_inp_coarse'] = self.input_imgs
        feed_dict['rois_inp'] = self.rois
        feed_dict['location_inp'] = self.object_locations
        feed_dict['class_inp'] = self.object_classes
        feed_dict['spatial_image'] = self.spatial_image
        
        all_predictions, _ = self.model.forward(feed_dict)
        ind = 0
        self.bIndices_pairs = []
        self.bIndices = suncg_parse.batchify(self.rois.data[:,0], self.rois.data[:,0])
        self.bIndices_pairs = torch.cat([a.repeat(a.size(0)) for a in self.bIndices]).view(-1,1)
        self.codes_pred = all_predictions['codes_pred']

        self.relative_trans_predictions = None
        self.relative_predictions = None
        if opts.pred_relative:
            self.relative_predictions = all_predictions['codes_relative']
        
        class_pred = None
        if opts.pred_class:
            class_pred = all_predictions['class_pred']

        common_class_predictions = None
        if opts.common_pred_class:
            common_class_predictions = all_predictions['common_class_pred']

        self.quat_pred_batch = None
        self.quat_gt_batch = None
    
        self.total_loss = 0
        self.loss_factors = {}
        
        # Siamese loss part
        id_pred = self.codes_pred['id']
        
        """
        loss_affinity = loss_utils.affinity_loss(id_pred, self.im2obj, 
                self.affinity_gt, opts.max_rois, loss_type=opts.id_loss_type, 
                class_pred=self.class_gt)
        self.loss_per_batch.append(loss_affinity.cpu().item())
        self.total_loss = loss_affinity
        self.loss_factors['affinity'] = loss_affinity
        """
        
        corr, total = affinity_eval(id_pred, self.im2obj, 
                self.affinity_gt, opts.max_rois, loss_type=opts.id_loss_type, 
                class_pred=self.class_gt)
        self.corr += corr
        self.total += total
        loss_affinity = 0
        self.total_loss = loss_affinity
        self.loss_factors['affinity'] = loss_affinity
        return
        
    
    def test(self):
        #print("debug {}".format(len(self.dataloader)))
        #total = len(self.dataloader)
        total = 500
        for i, batch in tqdm(enumerate(self.dataloader), total=total):
            self.set_input(batch)
            self.vis_iter = i
            if self.invalid_batch:
                continue
            self.forward()
            
            if i > total:
                break
            
        print("eval done")
        #mean_loss_affinity = np.array(self.loss_per_batch).mean()
        #print("mean loss affinity: {}".format(mean_loss_affinity))
        accu = self.corr / self.total
        print("accuracy: {}".format(accu))

def main(_):
    seed = 206
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.rendering_dir = osp.join(FLAGS.cache_dir, 'rendering')
    FLAGS.checkpoint_dir = osp.join(FLAGS.cache_dir, 'snapshots')
    if not FLAGS.classify_rot:
        FLAGS.nz_rot = 4

    if not FLAGS.classify_dir:
        FLAGS.nz_rel_dir = 3

    # FLAGS.n_data_workers = 0 # code crashes otherwise due to json not liking parallelization
    #trainer = SiameseTrainer(FLAGS)
    #trainer.init_training()
    #trainer.train()
    
    tester = SiameseTester(FLAGS)
    tester.define_model()
    tester.init_dataset()
    tester.test()


if __name__ == '__main__':
    app.run(main)
