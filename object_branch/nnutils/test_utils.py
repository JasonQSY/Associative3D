"""Generic Testing Utils.
"""
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags
import scipy.misc

from ..utils.visualizer import Visualizer

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_string('eval_set', 'val', 'which set to evaluate on')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')

flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')


## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Directory where networks are saved')
flags.DEFINE_string(
    'results_vis_dir', osp.join(cache_path, 'results_vis'),
    'Directory where intermittent results will be saved')

flags.DEFINE_string(
    'results_quality_dir', osp.join(cache_path, 'results_quality'),
    'Directory where intermittent results will be saved')

flags.DEFINE_string(
    'results_eval_dir', osp.join(cache_path, 'evaluation'),
    'Directory where evaluation results will be saved')

flags.DEFINE_boolean('save_visuals', False, 'Whether to save intermittent visuals')
flags.DEFINE_integer('visuals_freq', 50, 'Save visuals every few forward passes')
flags.DEFINE_integer('max_eval_iter', 0, 'Maximum evaluation iterations. 0 => 1 epoch.')

#-------- tranining class ---------#
#----------------------------------#
class Tester():
    def __init__(self, opts):
        self.opts = opts
        self.vis_iter = 0
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts_testing.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None, strict=True):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path), strict=strict)
        print('Loading weights from {}'.format(save_path))
        return

    def save_current_visuals(self):
        imgs_dir = osp.join(self.opts.results_vis_dir, 'vis_iter_{}'.format(self.vis_iter))
        if osp.exists(imgs_dir) and False:
            return
        else:
            visuals = self.get_current_visuals()
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)
            for k in visuals:
                img_path = osp.join(imgs_dir, k + '.png')
                scipy.misc.imsave(img_path, visuals[k])
    
    def homogenize_coordinates(self, locations):
        '''
        :param locations: N x 3 array of locations
        :return: homogeneous coordinates
        '''
        ones = torch.ones((locations.size(0), 1))
        homogenous_location = torch.cat([locations, ones], dim=1)
        return homogenous_location


    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_testing(self):
        opts = self.opts
        self.define_model()
        self.init_dataset()

    def test(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError
