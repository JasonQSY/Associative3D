"""Generic Training Utils.
"""
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags

from ..utils.visualizer import Visualizer
from ..utils.tb_visualizer import TBVisualizer
from ..optim import adam as ams_adam
from collections import Counter
#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')

flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_integer('num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')

## Flags for logging and snapshotting
# flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
#                     'Root directory for output files')
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 10000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 1, 'save model every k epochs')

## Flags for visualization
flags.DEFINE_integer('display_freq', 100, 'visuals logging frequency')
flags.DEFINE_boolean('display_visuals', False, 'whether to display images')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', False, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_boolean('use_html', False, 'Save html visualizations')
flags.DEFINE_integer('display_id', 1, 'Display Id')
flags.DEFINE_string('env_name', 'main', 'env name for experiments')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_integer('display_port', 8097, 'Display port')
flags.DEFINE_integer('display_single_pane_ncols', 0, 'if positive, display all images in a single visdom web panel with certain number of images per row.')
flags.DEFINE_boolean('tb', False, 'use tensorboard?')
flags.DEFINE_boolean('visdom', False, 'use visdom?')
flags.DEFINE_boolean('only_pairs', True, 'Train with only more than 2 examples per ')

#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        self.train_mode = True
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda()
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None, strict=True):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path), strict=strict)
        print('Loading weights from {}'.format(save_path))
        return

    def homogenize_coordinates(self, locations):
        '''
        :param locations: N x 3 array of locations
        :return: homogeneous coordinates
        '''
        ones = torch.ones((locations.size(0), 1))
        homogenous_location = torch.cat([locations, ones], dim=1)
        return homogenous_location

    def count_number_pairs(self, rois):
        counts = Counter(rois[:,0].numpy().tolist())
        pairs = sum([v*v for (k,v) in counts.items() if v > 1])
        return pairs

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_parameters(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.define_model()
        self.init_dataset()
        self.define_criterion()
        """
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.model.code_predictor.id_predictor.parameters(), 'lr': opts.learning_rate}
            ],
            lr=opts.learning_rate * 0.1, betas=(opts.beta1, 0.999)
        )
        """
        #self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=opts.learning_rate, momentum=0.9)
        self.optimizer = ams_adam.Adam(
            self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))

    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        if opts.tb:
            self.tb_visualizer = TBVisualizer(opts)
            # self.model.tb_visualizer  = self.tb_visualizer
            tb_visualizer = self.tb_visualizer

        total_steps = 0
        dataset_size = len(self.dataloader)
        start_time = time.time()
        opts.save_epoch_freq = min(opts.save_epoch_freq, opts.num_epochs - opts.num_pretrain_epochs)
        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):    
            epoch_iter = 0
            if opts.auto_rel_opt > 0 and opts.auto_rel_opt <= epoch:
                print('********* Swtiching automatic relative optimzation for epoch {} ****************'.format(epoch))
                opts.rel_opt=True
            for i, batch in enumerate(self.dataloader):
                iter_start_time = time.time()
                self.set_input(batch)
                if not self.invalid_batch:
                    self.optimizer.zero_grad()
                    self.forward()
                    self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.total_loss.item()
                    if self.train_mode:
                        self.total_loss.backward()
                        self.optimizer.step()

                total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    if opts.visdom:
                        visualizer.display_current_results(self.get_current_visuals(), epoch)
                        visualizer.plot_current_points(self.get_current_points())
                    if opts.tb:
                        #tb_visualizer.display_current_results(self.get_current_visuals(), total_steps)
                        tb_visualizer.plot_current_points(self.get_current_points(), total_steps)


                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    time_diff = time.time() - start_time
                    scalars_print = {k : v for k,v in scalars.items() if v != 0.0 }
                    visualizer.print_current_scalars(time_diff, epoch, epoch_iter, scalars_print)
                    if opts.plot_scalars:
                        if opts.visdom:
                            visualizer.plot_current_scalars(epoch, float(epoch_iter)/dataset_size, opts, scalars)
                        if opts.tb:
                            # tb_visualizer.log_histogram(total_steps, {'trans/pred' : self.codes_pred[3], 'trans/gt': self.codes_gt[3]})
                            tb_visualizer.plot_current_scalars(total_steps, opts, scalars)



                if total_steps % opts.save_latest_freq == 0:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return
            
            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)

        self.save('latest')
        self.save(opts.num_epochs)

