#import tensorflow as tf
#from torch.autograd import Variable
import numpy as np
import scipy.misc 
import os
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO


class Logger(object):
    """
    This is just a wrapper for torch.utils.tensorboard.
    """
    def __init__(self, log_dir, name=None):
        print("log_dir: {}".format(log_dir))
        self.writer = SummaryWriter(log_dir, filename_suffix=name)
    
    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, global_step=step)
    
    def image_summary(self, tag, images, step):
        self.writer.add_image(tag, images, global_step=step)
    
    def histo_summary(self, tag, values, step, bins=1000):
        self.writer.add_histogram(tag, values, global_step=step)

    def to_np(self, x):
        pass

    def to_var(self, x):
        pass

    def model_param_histo_summary(self, model, step):
        pass


"""
class Logger(object):
    
    def __init__(self, log_dir, name=None):
        # Create a summary writer logging to log_dir.
        if name is None:
            name = 'temp'
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            self.writer = tf.summary.FileWriter(os.path.join(log_dir, name),
                                                filename_suffix=name)
        else:
            self.writer = tf.summary.FileWriter(log_dir, filename_suffix=name)

    def scalar_summary(self, tag, value, step):
        # Log a scalar variable.
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        # Log a list of images.

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        # Log a histogram of the tensor of values.

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def to_np(self, x):
        return x.data.cpu().numpy()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def model_param_histo_summary(self, model, step):
        # log histogram summary of model's parameters and parameter gradients
        for tag, value in model.named_parameters():
            if value.grad is None:
            	continue
            tag = tag.replace('.', '/')
            tag = self.name+'/'+tag
            self.histo_summary(tag, self.to_np(value), step)
            self.histo_summary(tag+'/grad', self.to_np(value.grad), step)  
"""
