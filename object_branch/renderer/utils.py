import numpy as np
import os
import torch
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.offline as offline
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from torch.autograd import Variable
from ..utils import transformations
import cv2
from matplotlib import cm


curr_path = os.path.dirname(os.path.abspath(__file__))
blender_dir = os.path.join(os.path.dirname(os.path.dirname(curr_path)), 'blender')
blender_prefix = 'xvfb-run -n 0 blender -b --python '

colormap = np.array([[0.000000, 0.000000, 0.515625], [0.000000, 0.000000, 0.531250], [0.000000, 0.000000, 0.546875],
                     [0.000000, 0.000000, 0.562500], [0.000000, 0.000000, 0.578125], [0.000000, 0.000000, 0.593750],
                     [0.000000, 0.000000, 0.609375], [0.000000, 0.000000, 0.625000], [0.000000, 0.000000, 0.640625],
                     [0.000000, 0.000000, 0.656250], [0.000000, 0.000000, 0.671875], [0.000000, 0.000000, 0.687500],
                     [0.000000, 0.000000, 0.703125], [0.000000, 0.000000, 0.718750], [0.000000, 0.000000, 0.734375],
                     [0.000000, 0.000000, 0.750000], [0.000000, 0.000000, 0.765625], [0.000000, 0.000000, 0.781250],
                     [0.000000, 0.000000, 0.796875], [0.000000, 0.000000, 0.812500], [0.000000, 0.000000, 0.828125],
                     [0.000000, 0.000000, 0.843750], [0.000000, 0.000000, 0.859375], [0.000000, 0.000000, 0.875000],
                     [0.000000, 0.000000, 0.890625], [0.000000, 0.000000, 0.906250], [0.000000, 0.000000, 0.921875],
                     [0.000000, 0.000000, 0.937500], [0.000000, 0.000000, 0.953125], [0.000000, 0.000000, 0.968750],
                     [0.000000, 0.000000, 0.984375], [0.000000, 0.000000, 1.000000], [0.000000, 0.015625, 1.000000],
                     [0.000000, 0.031250, 1.000000], [0.000000, 0.046875, 1.000000], [0.000000, 0.062500, 1.000000],
                     [0.000000, 0.078125, 1.000000], [0.000000, 0.093750, 1.000000], [0.000000, 0.109375, 1.000000],
                     [0.000000, 0.125000, 1.000000], [0.000000, 0.140625, 1.000000], [0.000000, 0.156250, 1.000000],
                     [0.000000, 0.171875, 1.000000], [0.000000, 0.187500, 1.000000], [0.000000, 0.203125, 1.000000],
                     [0.000000, 0.218750, 1.000000], [0.000000, 0.234375, 1.000000], [0.000000, 0.250000, 1.000000],
                     [0.000000, 0.265625, 1.000000], [0.000000, 0.281250, 1.000000], [0.000000, 0.296875, 1.000000],
                     [0.000000, 0.312500, 1.000000], [0.000000, 0.328125, 1.000000], [0.000000, 0.343750, 1.000000],
                     [0.000000, 0.359375, 1.000000], [0.000000, 0.375000, 1.000000], [0.000000, 0.390625, 1.000000],
                     [0.000000, 0.406250, 1.000000], [0.000000, 0.421875, 1.000000], [0.000000, 0.437500, 1.000000],
                     [0.000000, 0.453125, 1.000000], [0.000000, 0.468750, 1.000000], [0.000000, 0.484375, 1.000000],
                     [0.000000, 0.500000, 1.000000], [0.000000, 0.515625, 1.000000], [0.000000, 0.531250, 1.000000],
                     [0.000000, 0.546875, 1.000000], [0.000000, 0.562500, 1.000000], [0.000000, 0.578125, 1.000000],
                     [0.000000, 0.593750, 1.000000], [0.000000, 0.609375, 1.000000], [0.000000, 0.625000, 1.000000],
                     [0.000000, 0.640625, 1.000000], [0.000000, 0.656250, 1.000000], [0.000000, 0.671875, 1.000000],
                     [0.000000, 0.687500, 1.000000], [0.000000, 0.703125, 1.000000], [0.000000, 0.718750, 1.000000],
                     [0.000000, 0.734375, 1.000000], [0.000000, 0.750000, 1.000000], [0.000000, 0.765625, 1.000000],
                     [0.000000, 0.781250, 1.000000], [0.000000, 0.796875, 1.000000], [0.000000, 0.812500, 1.000000],
                     [0.000000, 0.828125, 1.000000], [0.000000, 0.843750, 1.000000], [0.000000, 0.859375, 1.000000],
                     [0.000000, 0.875000, 1.000000], [0.000000, 0.890625, 1.000000], [0.000000, 0.906250, 1.000000],
                     [0.000000, 0.921875, 1.000000], [0.000000, 0.937500, 1.000000], [0.000000, 0.953125, 1.000000],
                     [0.000000, 0.968750, 1.000000], [0.000000, 0.984375, 1.000000], [0.000000, 1.000000, 1.000000],
                     [0.015625, 1.000000, 0.984375], [0.031250, 1.000000, 0.968750], [0.046875, 1.000000, 0.953125],
                     [0.062500, 1.000000, 0.937500], [0.078125, 1.000000, 0.921875], [0.093750, 1.000000, 0.906250],
                     [0.109375, 1.000000, 0.890625], [0.125000, 1.000000, 0.875000], [0.140625, 1.000000, 0.859375],
                     [0.156250, 1.000000, 0.843750], [0.171875, 1.000000, 0.828125], [0.187500, 1.000000, 0.812500],
                     [0.203125, 1.000000, 0.796875], [0.218750, 1.000000, 0.781250], [0.234375, 1.000000, 0.765625],
                     [0.250000, 1.000000, 0.750000], [0.265625, 1.000000, 0.734375], [0.281250, 1.000000, 0.718750],
                     [0.296875, 1.000000, 0.703125], [0.312500, 1.000000, 0.687500], [0.328125, 1.000000, 0.671875],
                     [0.343750, 1.000000, 0.656250], [0.359375, 1.000000, 0.640625], [0.375000, 1.000000, 0.625000],
                     [0.390625, 1.000000, 0.609375], [0.406250, 1.000000, 0.593750], [0.421875, 1.000000, 0.578125],
                     [0.437500, 1.000000, 0.562500], [0.453125, 1.000000, 0.546875], [0.468750, 1.000000, 0.531250],
                     [0.484375, 1.000000, 0.515625], [0.500000, 1.000000, 0.500000], [0.515625, 1.000000, 0.484375],
                     [0.531250, 1.000000, 0.468750], [0.546875, 1.000000, 0.453125], [0.562500, 1.000000, 0.437500],
                     [0.578125, 1.000000, 0.421875], [0.593750, 1.000000, 0.406250], [0.609375, 1.000000, 0.390625],
                     [0.625000, 1.000000, 0.375000], [0.640625, 1.000000, 0.359375], [0.656250, 1.000000, 0.343750],
                     [0.671875, 1.000000, 0.328125], [0.687500, 1.000000, 0.312500], [0.703125, 1.000000, 0.296875],
                     [0.718750, 1.000000, 0.281250], [0.734375, 1.000000, 0.265625], [0.750000, 1.000000, 0.250000],
                     [0.765625, 1.000000, 0.234375], [0.781250, 1.000000, 0.218750], [0.796875, 1.000000, 0.203125],
                     [0.812500, 1.000000, 0.187500], [0.828125, 1.000000, 0.171875], [0.843750, 1.000000, 0.156250],
                     [0.859375, 1.000000, 0.140625], [0.875000, 1.000000, 0.125000], [0.890625, 1.000000, 0.109375],
                     [0.906250, 1.000000, 0.093750], [0.921875, 1.000000, 0.078125], [0.937500, 1.000000, 0.062500],
                     [0.953125, 1.000000, 0.046875], [0.968750, 1.000000, 0.031250], [0.984375, 1.000000, 0.015625],
                     [1.000000, 1.000000, 0.000000], [1.000000, 0.984375, 0.000000], [1.000000, 0.968750, 0.000000],
                     [1.000000, 0.953125, 0.000000], [1.000000, 0.937500, 0.000000], [1.000000, 0.921875, 0.000000],
                     [1.000000, 0.906250, 0.000000], [1.000000, 0.890625, 0.000000], [1.000000, 0.875000, 0.000000],
                     [1.000000, 0.859375, 0.000000], [1.000000, 0.843750, 0.000000], [1.000000, 0.828125, 0.000000],
                     [1.000000, 0.812500, 0.000000], [1.000000, 0.796875, 0.000000], [1.000000, 0.781250, 0.000000],
                     [1.000000, 0.765625, 0.000000], [1.000000, 0.750000, 0.000000], [1.000000, 0.734375, 0.000000],
                     [1.000000, 0.718750, 0.000000], [1.000000, 0.703125, 0.000000], [1.000000, 0.687500, 0.000000],
                     [1.000000, 0.671875, 0.000000], [1.000000, 0.656250, 0.000000], [1.000000, 0.640625, 0.000000],
                     [1.000000, 0.625000, 0.000000], [1.000000, 0.609375, 0.000000], [1.000000, 0.593750, 0.000000],
                     [1.000000, 0.578125, 0.000000], [1.000000, 0.562500, 0.000000], [1.000000, 0.546875, 0.000000],
                     [1.000000, 0.531250, 0.000000], [1.000000, 0.515625, 0.000000], [1.000000, 0.500000, 0.000000],
                     [1.000000, 0.484375, 0.000000], [1.000000, 0.468750, 0.000000], [1.000000, 0.453125, 0.000000],
                     [1.000000, 0.437500, 0.000000], [1.000000, 0.421875, 0.000000], [1.000000, 0.406250, 0.000000],
                     [1.000000, 0.390625, 0.000000], [1.000000, 0.375000, 0.000000], [1.000000, 0.359375, 0.000000],
                     [1.000000, 0.343750, 0.000000], [1.000000, 0.328125, 0.000000], [1.000000, 0.312500, 0.000000],
                     [1.000000, 0.296875, 0.000000], [1.000000, 0.281250, 0.000000], [1.000000, 0.265625, 0.000000],
                     [1.000000, 0.250000, 0.000000], [1.000000, 0.234375, 0.000000], [1.000000, 0.218750, 0.000000],
                     [1.000000, 0.203125, 0.000000], [1.000000, 0.187500, 0.000000], [1.000000, 0.171875, 0.000000],
                     [1.000000, 0.156250, 0.000000], [1.000000, 0.140625, 0.000000], [1.000000, 0.125000, 0.000000],
                     [1.000000, 0.109375, 0.000000], [1.000000, 0.093750, 0.000000], [1.000000, 0.078125, 0.000000],
                     [1.000000, 0.062500, 0.000000], [1.000000, 0.046875, 0.000000], [1.000000, 0.031250, 0.000000],
                     [1.000000, 0.015625, 0.000000], [1.000000, 0.000000, 0.000000], [0.984375, 0.000000, 0.000000],
                     [0.968750, 0.000000, 0.000000], [0.953125, 0.000000, 0.000000], [0.937500, 0.000000, 0.000000],
                     [0.921875, 0.000000, 0.000000], [0.906250, 0.000000, 0.000000], [0.890625, 0.000000, 0.000000],
                     [0.875000, 0.000000, 0.000000], [0.859375, 0.000000, 0.000000], [0.843750, 0.000000, 0.000000],
                     [0.828125, 0.000000, 0.000000], [0.812500, 0.000000, 0.000000], [0.796875, 0.000000, 0.000000],
                     [0.781250, 0.000000, 0.000000], [0.765625, 0.000000, 0.000000], [0.750000, 0.000000, 0.000000],
                     [0.734375, 0.000000, 0.000000], [0.718750, 0.000000, 0.000000], [0.703125, 0.000000, 0.000000],
                     [0.687500, 0.000000, 0.000000], [0.671875, 0.000000, 0.000000], [0.656250, 0.000000, 0.000000],
                     [0.640625, 0.000000, 0.000000], [0.625000, 0.000000, 0.000000], [0.609375, 0.000000, 0.000000],
                     [0.593750, 0.000000, 0.000000], [0.578125, 0.000000, 0.000000], [0.562500, 0.000000, 0.000000],
                     [0.546875, 0.000000, 0.000000], [0.531250, 0.000000, 0.000000], [0.515625, 0.000000, 0.000000],
                     [0.500000, 0.000000, 0.000000]])

"""
blue_colormap = np.array([[0.000000, 0.000000, 0.515625], [0.000000, 0.000000, 0.531250], [0.000000, 0.000000, 0.546875],
                     [0.000000, 0.000000, 0.562500], [0.000000, 0.000000, 0.578125], [0.000000, 0.000000, 0.593750],
                     [0.000000, 0.000000, 0.609375], [0.000000, 0.000000, 0.625000], [0.000000, 0.000000, 0.640625],
                     [0.000000, 0.000000, 0.656250], [0.000000, 0.000000, 0.671875], [0.000000, 0.000000, 0.687500],
                     [0.000000, 0.000000, 0.703125], [0.000000, 0.000000, 0.718750], [0.000000, 0.000000, 0.734375],
                     [0.000000, 0.000000, 0.750000], [0.000000, 0.000000, 0.765625], [0.000000, 0.000000, 0.781250],
                     [0.000000, 0.000000, 0.796875], [0.000000, 0.000000, 0.812500], [0.000000, 0.000000, 0.828125],
                     [0.000000, 0.000000, 0.843750], [0.000000, 0.000000, 0.859375], [0.000000, 0.000000, 0.875000],
                     [0.000000, 0.000000, 0.890625], [0.000000, 0.000000, 0.906250], [0.000000, 0.000000, 0.921875],
                     [0.000000, 0.000000, 0.937500], [0.000000, 0.000000, 0.953125], [0.000000, 0.000000, 0.968750],
                     [0.000000, 0.000000, 0.984375], [0.000000, 0.000000, 1.000000]])

red_colormap = np.array([[1.000000, 0.000000, 0.000000], [0.984375, 0.000000, 0.000000],
                     [0.968750, 0.000000, 0.000000], [0.953125, 0.000000, 0.000000], [0.937500, 0.000000, 0.000000],
                     [0.921875, 0.000000, 0.000000], [0.906250, 0.000000, 0.000000], [0.890625, 0.000000, 0.000000],
                     [0.875000, 0.000000, 0.000000], [0.859375, 0.000000, 0.000000], [0.843750, 0.000000, 0.000000],
                     [0.828125, 0.000000, 0.000000], [0.812500, 0.000000, 0.000000], [0.796875, 0.000000, 0.000000],
                     [0.781250, 0.000000, 0.000000], [0.765625, 0.000000, 0.000000], [0.750000, 0.000000, 0.000000],
                     [0.734375, 0.000000, 0.000000], [0.718750, 0.000000, 0.000000], [0.703125, 0.000000, 0.000000],
                     [0.687500, 0.000000, 0.000000], [0.671875, 0.000000, 0.000000], [0.656250, 0.000000, 0.000000],
                     [0.640625, 0.000000, 0.000000], [0.625000, 0.000000, 0.000000], [0.609375, 0.000000, 0.000000],
                     [0.593750, 0.000000, 0.000000], [0.578125, 0.000000, 0.000000], [0.562500, 0.000000, 0.000000],
                     [0.546875, 0.000000, 0.000000], [0.531250, 0.000000, 0.000000], [0.515625, 0.000000, 0.000000],
                     [0.500000, 0.000000, 0.000000]])

green_colormap = np.array([[0.500000, 1.000000, 0.500000], [0.515625, 1.000000, 0.484375],
                     [0.531250, 1.000000, 0.468750], [0.546875, 1.000000, 0.453125], [0.562500, 1.000000, 0.437500],
                     [0.578125, 1.000000, 0.421875], [0.593750, 1.000000, 0.406250], [0.609375, 1.000000, 0.390625],
                     [0.625000, 1.000000, 0.375000], [0.640625, 1.000000, 0.359375], [0.656250, 1.000000, 0.343750],
                     [0.671875, 1.000000, 0.328125], [0.687500, 1.000000, 0.312500], [0.703125, 1.000000, 0.296875],
                     [0.718750, 1.000000, 0.281250], [0.734375, 1.000000, 0.265625], [0.750000, 1.000000, 0.250000],
                     [0.765625, 1.000000, 0.234375], [0.781250, 1.000000, 0.218750], [0.796875, 1.000000, 0.203125],
                     [0.812500, 1.000000, 0.187500], [0.828125, 1.000000, 0.171875], [0.843750, 1.000000, 0.156250],
                     [0.859375, 1.000000, 0.140625], [0.875000, 1.000000, 0.125000], [0.890625, 1.000000, 0.109375],
                     [0.906250, 1.000000, 0.093750], [0.921875, 1.000000, 0.078125], [0.937500, 1.000000, 0.062500],
                     [0.953125, 1.000000, 0.046875], [0.968750, 1.000000, 0.031250], [0.984375, 1.000000, 0.015625],
                     [1.000000, 1.000000, 0.000000]])
"""
blue_colormap = np.array([[0.000000, 0.000000, 0.515625], [0.000000, 0.000000, 0.531250], [0.000000, 0.000000, 0.546875],
                     [0.000000, 0.000000, 0.562500], [0.000000, 0.000000, 0.578125], [0.000000, 0.000000, 0.593750],
                     [0.000000, 0.000000, 0.609375], [0.000000, 0.000000, 0.625000], [0.000000, 0.000000, 0.640625],
                     [0.000000, 0.000000, 0.656250], [0.000000, 0.000000, 0.671875], [0.000000, 0.000000, 0.687500],
                     [0.000000, 0.000000, 0.703125], [0.000000, 0.000000, 0.718750], [0.000000, 0.000000, 0.734375],
                     [0.000000, 0.000000, 0.750000], [0.000000, 0.000000, 0.765625], [0.000000, 0.000000, 0.781250],
                     [0.000000, 0.000000, 0.796875], [0.000000, 0.000000, 0.812500], [0.000000, 0.000000, 0.828125],
                     [0.000000, 0.000000, 0.843750], [0.000000, 0.000000, 0.859375], [0.000000, 0.000000, 0.875000],
                     [0.000000, 0.000000, 0.890625], [0.000000, 0.000000, 0.906250], [0.000000, 0.000000, 0.921875],
                     [0.000000, 0.000000, 0.937500], [0.000000, 0.000000, 0.953125], [0.000000, 0.000000, 0.968750],
                     [0.000000, 0.000000, 0.984375], [0.000000, 0.000000, 1.000000], [0.000000, 0.015625, 1.000000],
                     [0.000000, 0.031250, 1.000000], [0.000000, 0.046875, 1.000000], [0.000000, 0.062500, 1.000000],
                     [0.000000, 0.078125, 1.000000], [0.000000, 0.093750, 1.000000], [0.000000, 0.109375, 1.000000],
                     [0.000000, 0.125000, 1.000000], [0.000000, 0.140625, 1.000000], [0.000000, 0.156250, 1.000000],
                     [0.000000, 0.171875, 1.000000], [0.000000, 0.187500, 1.000000], [0.000000, 0.203125, 1.000000],
                     [0.000000, 0.218750, 1.000000], [0.000000, 0.234375, 1.000000], [0.000000, 0.250000, 1.000000],
                     [0.000000, 0.265625, 1.000000], [0.000000, 0.281250, 1.000000], [0.000000, 0.296875, 1.000000],
                     [0.000000, 0.312500, 1.000000], [0.000000, 0.328125, 1.000000], [0.000000, 0.343750, 1.000000],
                     [0.000000, 0.359375, 1.000000], [0.000000, 0.375000, 1.000000], [0.000000, 0.390625, 1.000000],
                     [0.000000, 0.406250, 1.000000], [0.000000, 0.421875, 1.000000], [0.000000, 0.437500, 1.000000],
                     [0.000000, 0.453125, 1.000000], [0.000000, 0.468750, 1.000000], [0.000000, 0.484375, 1.000000],
                     [0.000000, 0.500000, 1.000000]])


green_colormap = np.array([[0.000000, 0.515625, 0.000000], [0.000000, 0.531250, 0.000000],
                     [0.000000, 0.546875, 0.000000], [0.000000, 0.562500, 0.000000], [0.000000, 0.578125, 0.000000],
                     [0.000000, 0.593750, 0.000000], [0.000000, 0.609375, 0.000000], [0.000000, 0.625000, 0.000000],
                     [0.000000, 0.640625, 0.000000], [0.000000, 0.656250, 0.000000], [0.000000, 0.671875, 0.000000],
                     [0.000000, 0.687500, 0.000000], [0.000000, 0.703125, 0.000000], [0.000000, 0.718750, 0.000000],
                     [0.000000, 0.734375, 0.000000], [0.000000, 0.750000, 0.000000], [0.000000, 0.765625, 0.000000],
                     [0.000000, 0.781250, 0.000000], [0.000000, 0.796875, 0.000000], [0.000000, 0.812500, 0.000000],
                     [0.000000, 0.828125, 0.000000], [0.000000, 0.843750, 0.000000], [0.000000, 0.859375, 0.000000],
                     [0.000000, 0.875000, 0.000000], [0.000000, 0.890625, 0.000000], [0.000000, 0.906250, 0.000000],
                     [0.000000, 0.921875, 0.000000], [0.000000, 0.937500, 0.000000], [0.000000, 0.953125, 0.000000],
                     [0.000000, 0.968750, 0.000000], [0.000000, 0.984375, 0.000000], [0.000000, 1.000000, 0.000000],
                     [0.015625, 1.000000, 0.000000], [0.031250, 1.000000, 0.000000], [0.046875, 1.000000, 0.000000],
                     [0.062500, 1.000000, 0.000000], [0.078125, 1.000000, 0.000000], [0.093750, 1.000000, 0.000000],
                     [0.109375, 1.000000, 0.000000], [0.125000, 1.000000, 0.000000], [0.140625, 1.000000, 0.000000],
                     [0.156250, 1.000000, 0.000000], [0.171875, 1.000000, 0.000000], [0.187500, 1.000000, 0.000000],
                     [0.203125, 1.000000, 0.000000], [0.218750, 1.000000, 0.000000], [0.234375, 1.000000, 0.000000],
                     [0.250000, 1.000000, 0.000000], [0.265625, 1.000000, 0.000000], [0.281250, 1.000000, 0.000000],
                     [0.296875, 1.000000, 0.000000], [0.312500, 1.000000, 0.000000], [0.328125, 1.000000, 0.000000],
                     [0.343750, 1.000000, 0.000000], [0.359375, 1.000000, 0.000000], [0.375000, 1.000000, 0.000000],
                     [0.390625, 1.000000, 0.000000], [0.406250, 1.000000, 0.000000], [0.421875, 1.000000, 0.000000],
                     [0.437500, 1.000000, 0.000000], [0.453125, 1.000000, 0.000000], [0.468750, 1.000000, 0.000000],
                     [0.484375, 1.000000, 0.000000], [0.500000, 1.000000, 0.000000], [0.515625, 1.000000, 0.000000],
                     [0.531250, 0.531250, 0.000000],
                     [0.546875, 1.000000, 0.000000], [0.562500, 1.000000, 0.000000], [0.578125, 1.000000, 0.000000],
                     [0.593750, 1.000000, 0.000000], [0.609375, 1.000000, 0.000000], [0.625000, 1.000000, 0.000000],
                     [0.640625, 1.000000, 0.000000], [0.656250, 1.000000, 0.000000], [0.671875, 1.000000, 0.000000],
                     [0.687500, 1.000000, 0.000000], [0.703125, 1.000000, 0.000000], [0.718750, 1.000000, 0.000000],
                     [0.734375, 1.000000, 0.000000], [0.750000, 1.000000, 0.000000], [0.765625, 1.000000, 0.000000],
                     [0.781250, 1.000000, 0.000000], [0.796875, 1.000000, 0.000000], [0.812500, 1.000000, 0.000000],
                     [0.828125, 1.000000, 0.000000], [0.843750, 1.000000, 0.000000], [0.859375, 1.000000, 0.000000],
                     [0.875000, 1.000000, 0.000000], [0.890625, 1.000000, 0.000000], [0.906250, 1.000000, 0.000000],
                     [0.921875, 1.000000, 0.000000], [0.937500, 1.000000, 0.000000], [0.953125, 1.000000, 0.000000],
                     [0.968750, 1.000000, 0.000000], [0.984375, 1.000000, 0.000000], [1.000000, 1.000000, 0.000000]
                     ])


red_colormap = np.array([[1.000000, 0.500000, 0.000000],
                     [1.000000, 0.484375, 0.000000], [1.000000, 0.468750, 0.000000], [1.000000, 0.453125, 0.000000],
                     [1.000000, 0.437500, 0.000000], [1.000000, 0.421875, 0.000000], [1.000000, 0.406250, 0.000000],
                     [1.000000, 0.390625, 0.000000], [1.000000, 0.375000, 0.000000], [1.000000, 0.359375, 0.000000],
                     [1.000000, 0.343750, 0.000000], [1.000000, 0.328125, 0.000000], [1.000000, 0.312500, 0.000000],
                     [1.000000, 0.296875, 0.000000], [1.000000, 0.281250, 0.000000], [1.000000, 0.265625, 0.000000],
                     [1.000000, 0.250000, 0.000000], [1.000000, 0.234375, 0.000000], [1.000000, 0.218750, 0.000000],
                     [1.000000, 0.203125, 0.000000], [1.000000, 0.187500, 0.000000], [1.000000, 0.171875, 0.000000],
                     [1.000000, 0.156250, 0.000000], [1.000000, 0.140625, 0.000000], [1.000000, 0.125000, 0.000000],
                     [1.000000, 0.109375, 0.000000], [1.000000, 0.093750, 0.000000], [1.000000, 0.078125, 0.000000],
                     [1.000000, 0.062500, 0.000000], [1.000000, 0.046875, 0.000000], [1.000000, 0.031250, 0.000000],
                     [1.000000, 0.015625, 0.000000], [1.000000, 0.000000, 0.000000], [0.984375, 0.000000, 0.000000],
                     [0.968750, 0.000000, 0.000000], [0.953125, 0.000000, 0.000000], [0.937500, 0.000000, 0.000000],
                     [0.921875, 0.000000, 0.000000], [0.906250, 0.000000, 0.000000], [0.890625, 0.000000, 0.000000],
                     [0.875000, 0.000000, 0.000000], [0.859375, 0.000000, 0.000000], [0.843750, 0.000000, 0.000000],
                     [0.828125, 0.000000, 0.000000], [0.812500, 0.000000, 0.000000], [0.796875, 0.000000, 0.000000],
                     [0.781250, 0.000000, 0.000000], [0.765625, 0.000000, 0.000000], [0.750000, 0.000000, 0.000000],
                     [0.734375, 0.000000, 0.000000], [0.718750, 0.000000, 0.000000], [0.703125, 0.000000, 0.000000],
                     [0.687500, 0.000000, 0.000000], [0.671875, 0.000000, 0.000000], [0.656250, 0.000000, 0.000000],
                     [0.640625, 0.000000, 0.000000], [0.625000, 0.000000, 0.000000], [0.609375, 0.000000, 0.000000],
                     [0.593750, 0.000000, 0.000000], [0.578125, 0.000000, 0.000000], [0.562500, 0.000000, 0.000000],
                     [0.546875, 0.000000, 0.000000], [0.531250, 0.000000, 0.000000], [0.515625, 0.000000, 0.000000],
                     [0.500000, 0.000000, 0.000000]])

cube_v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
cube_v = cube_v - 0.5

cube_f = np.array(
    [[1, 7, 5], [1, 3, 7], [1, 4, 3], [1, 2, 4], [3, 8, 7], [3, 4, 8], [5, 7, 8], [5, 8, 6], [1, 5, 6], [1, 6, 2],
     [2, 6, 8], [2, 8, 4]]).astype(np.int)


def voxels_to_mesh(pred_vol, thresh=0.5):
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3).reshape(3, 12 * tot_points).transpose()
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    radius = pred_vol[x, y, z]
                    v_all[v_counter:v_counter + 8, :] *= radius
                    v_all[v_counter:v_counter + 8, :] += (np.array([[x, y, z]]) + 0.5)
                    v_counter += 8

    return v_all, f_all


def voxels_to_points(pred_vol, thresh=0.5):
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    v_all = np.zeros([tot_points, 3])
    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    v_all[v_counter, :] = (np.array([[x, y, z]]) + 0.5)
                    v_counter += 1
    return v_all


def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


def append_mtl_obj(mf_handle, vertices, faces, mtl_ids):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('usemtl m{}\n'.format(mtl_ids[fx]))
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


def append_mtl(mtl_handle, mtl_ids, colors):
    for mx in range(len(mtl_ids)):
        mtl_handle.write('newmtl m{}\n'.format(mtl_ids[mx]))
        mtl_handle.write('Kd {:f} {:f} {:f}\n'.format(colors[mx, 0], colors[mx, 1], colors[mx, 2]))
        mtl_handle.write('Ka 0 0 0\n')
    return


def render_mesh(mesh_file, png_dir, scale=0.5):
    cmd = blender_prefix + '{:s}/render_script.py  --obj_file {:s} --out_dir {:s} --r 2 --delta_theta 30 --sz_x {} --sz_y {} > /dev/null'.format(
        blender_dir, mesh_file, png_dir, int(640 * scale), int(480 * scale))
    print(cmd)
    os.system(cmd)
    return


def render_directory(mesh_dir, png_dir, scale=0.5):
    cmd = blender_prefix + '{:s}/render_dir_script.py  --obj_dir {:s} --out_dir {:s} --r 2 --delta_theta 30 --sz_x {} --sz_y {} > /dev/null'.format(
        blender_dir, mesh_dir, png_dir, int(640 * scale), int(480 * scale))
    print(cmd)
    os.system(cmd)
    return


class Downsample(torch.nn.Module):
    def __init__(self, s, use_max=False, batch_mode=False):
        super(Downsample, self).__init__()
        self.batch_mode = batch_mode
        if (use_max):
            layer = torch.nn.MaxPool3d(s, stride=s)
        else:
            layer = torch.nn.Conv3d(1, 1, s, stride=s)
            layer.weight.data.fill_(1. / layer.weight.data.nelement())
            layer.bias.data.fill_(0)
        self.layer = layer

    def forward(self, vol):
        if self.batch_mode:
            out_vol = self.layer.forward(vol)
        else:
            out_vol = self.layer.forward(torch.unsqueeze(torch.unsqueeze(vol, 0), 0))[0, 0]
        return out_vol


def downsample(vol, s, use_max=False):
    module = Downsample(s, use_max=use_max).type(torch.FloatTensor)
    vol_var = Variable(torch.from_numpy(vol), requires_grad=False).type(torch.FloatTensor)
    return module.forward(vol_var).data.numpy()


def prediction_to_entity(pred):
    if torch.is_tensor(pred[0]):
        pred = [p.numpy() for p in pred]
    volume = pred[0].astype(np.float)
    transform = pred[1].astype(np.float)
    if transform.shape[0] == 4 and transform.shape[1] == 4:
        return volume, transform
    else:
        scale_mat = np.diag(pred[1].astype(np.float))
        rot_mat = transformations.quaternion_matrix(pred[2].astype(np.float))[0:3, 0:3]
        transform = np.eye(4)
        transform[0:3, 0:3] = np.matmul(rot_mat, scale_mat)
        transform[0:3, 3] = pred[3].astype(np.float)
        return volume, transform


def save_parse(mesh_file, codes, thresh=0.5, use_soft_voxels=True, save_objectwise=False):
    mtl_file = mesh_file.replace('.obj', '.mtl')
    fout_mtl = open(mtl_file, 'w')
    mtl_file = mtl_file.split('/')[-1]

    n_parts = len(codes)
    nA = 0
    nB = 0
    nM = 0
    if n_parts != 0 and len(codes[0]) == 5:
        for p in range(n_parts):
            cmap_str = codes[p][4]
            if cmap_str == 'red':
                nA += 1
            elif cmap_str == 'blue':
                nB += 1
            elif cmap_str == 'green':
                nM += 1
        color_r = np.linspace(0, red_colormap.shape[0] - 1, nA).astype(np.int).tolist()
        color_b = np.linspace(0, blue_colormap.shape[0] - 1, nB).astype(np.int).tolist()
        color_g = np.linspace(0, green_colormap.shape[0] - 1, nM).astype(np.int).tolist()
        r_idx = 0
        b_idx = 0
        g_idx = 0
    
    #color_inds = np.linspace(0, 255, n_parts).astype(np.int).tolist()
    #using_cmap = colormap
    for p in range(n_parts):
        if len(codes[p]) == 5:
            cmap_str = codes[p][4]
            if cmap_str == 'red':
                #color_ind = np.random.randint(0, red_colormap.shape[0])
                color_ind = color_r[r_idx]
                r_idx += 1
                cmap = red_colormap[color_ind]
            elif cmap_str == 'blue':
                #color_ind = np.random.randint(0, blue_colormap.shape[0])
                color_ind = color_b[b_idx]
                b_idx += 1
                cmap = blue_colormap[color_ind]
            elif cmap_str == 'green':
                #color_ind = np.random.randint(0, green_colormap.shape[0])
                color_ind = color_g[g_idx]
                g_idx += 1
                cmap = green_colormap[color_ind]
        else:
            color_ind = np.random.randint(0, 255)
            cmap = colormap[color_ind]

        #cmap = using_cmap[color_inds[p]]
        fout_mtl.write(
            'newmtl m{:d}\nKd {:f} {:f} {:f}\nKa 0 0 0\n'.format(p, cmap[0], cmap[1], cmap[2]))
    fout_mtl.close()
    if not save_objectwise:
        fout = open(mesh_file, 'w')
        fout.write('mtllib {:s}\n'.format(mtl_file))

    f_counter = 0
    for p in range(n_parts):
        volume, transform = prediction_to_entity(codes[p])
        if save_objectwise:
            fout = open(mesh_file.replace('.obj', '_' + str(p) + '.obj'), 'w')
            fout.write('mtllib {:s}\n'.format(mtl_file))
        volume = downsample(volume, volume.shape[0] // 32)
        v, f = voxels_to_mesh(volume, thresh=thresh)
        v = v / 32 - 0.5

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()
            fout.write('usemtl m{:d}\n'.format(p))
            append_obj(fout, v_transformed, f + f_counter)

            if not save_objectwise:
                f_counter += n_verts

        if save_objectwise or p == (n_parts - 1):
            fout.close()
            
            
def save_parse_with_affinity(mesh_file, codes1, thresh=0.5, use_soft_voxels=True, save_objectwise=False):
    mtl_file = mesh_file.replace('.obj', '.mtl')
    fout_mtl = open(mtl_file, 'w')
    mtl_file = mtl_file.split('/')[-1]

    n_parts = len(codes)
    color_inds = np.linspace(0, 255, n_parts).astype(np.int).tolist()
    for p in range(n_parts):
        cmap = colormap[color_inds[p]]
        fout_mtl.write(
            'newmtl m{:d}\nKd {:f} {:f} {:f}\nKa 0 0 0\n'.format(p, cmap[0], cmap[1], cmap[2]))
    fout_mtl.close()
    if not save_objectwise:
        fout = open(mesh_file, 'w')
        fout.write('mtllib {:s}\n'.format(mtl_file))

    f_counter = 0
    for p in range(n_parts):
        volume, transform = prediction_to_entity(codes[p])
        if save_objectwise:
            fout = open(mesh_file.replace('.obj', '_' + str(p) + '.obj'), 'w')
            fout.write('mtllib {:s}\n'.format(mtl_file))
        volume = downsample(volume, volume.shape[0] // 32)
        v, f = voxels_to_mesh(volume, thresh=thresh)
        v = v / 32 - 0.5

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()
            fout.write('usemtl m{:d}\n'.format(p))
            append_obj(fout, v_transformed, f + f_counter)

            if not save_objectwise:
                f_counter += n_verts

        if save_objectwise or p == (n_parts - 1):
            fout.close()



def codes_to_points(codes, thresh=0.5, objectwise=False):
    scene_verts = []
    n_parts = len(codes)
    for p in range(n_parts):
        volume, transform = prediction_to_entity(codes[p])
        volume = downsample(volume, volume.shape[0] // 32)
        v = voxels_to_points(volume, thresh=thresh)
        v = v / 32 - 0.5

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()
            scene_verts.append(v_transformed)

    if not objectwise:
        scene_verts = np.concatenate(scene_verts, axis=0)

    return scene_verts


def dispmap_to_mesh(dmap, k_mat, scale_x=1, scale_y=1, min_disp=1e-2):
    '''
    Converts a inverse depth map to a 3D point cloud.
    
    Args:
        dmap: H X W inverse depth map
        k_mat : 3 X 3 intrinsic matrix
        scale_x: Scale the intrinsic matrix's x row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        scale_y: Scale the intrinsic matrix's y row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        min_disp: Points with disp less than this are not rendered
    Returns:
        vs: n_pts X 3 [x,y,z] coordinates
        fs: mesh faces
    '''
    H = np.shape(dmap)[0]
    W = np.shape(dmap)[1]
    dmap = dmap.reshape((H, W))
    k_mat[0, :] = scale_x * k_mat[0, :]
    k_mat[1, :] = scale_y * k_mat[1, :]
    k_inv = np.linalg.inv(k_mat)
    num_pts = H * W
    pts = np.ones((3, num_pts))
    ctr = 0
    for y in range(H):
        for x in range(W):
            pts[0, ctr] = x + 0.5
            pts[1, ctr] = y + 0.5
            pts[:, ctr] *= (1 / dmap[y, x])
            ctr += 1

    verts = np.transpose(np.matmul(k_inv, pts))
    num_faces_max = H * W * 2
    faces = np.zeros((num_faces_max, 3))
    face_ctr = 0
    for y in range(H - 1):
        for x in range(W - 1):
            if (dmap[y, x] > min_disp) and (dmap[y, x + 1] > min_disp) and (dmap[y + 1, x + 1] > min_disp):
                faces[face_ctr, 0] = y * W + x + 1
                faces[face_ctr, 1] = y * W + (x + 1) + 1
                faces[face_ctr, 2] = (y + 1) * W + (x + 1) + 1
                face_ctr += 1

            if (dmap[y, x] > min_disp) and (dmap[y + 1, x] > min_disp) and (dmap[y + 1, x + 1] > min_disp):
                faces[face_ctr, 0] = y * W + x + 1
                faces[face_ctr, 1] = (y + 1) * W + x + 1
                faces[face_ctr, 2] = (y + 1) * W + (x + 1) + 1
                face_ctr += 1
    faces = faces[0:face_ctr, :]
    return verts, faces.astype(np.int)


def dispmap_to_points(dmap, k_mat, scale_x=1, scale_y=1, min_disp=1e-2):
    '''
    Converts a inverse depth map to a 3D point cloud.
    
    Args:
        dmap: H X W inverse depth map
        k_mat : 3 X 3 intrinsic matrix
        scale_x: Scale the intrinsic matrix's x row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        scale_y: Scale the intrinsic matrix's y row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        min_disp: Points with disp less than this are not rendered
    Returns:
        n_pts X 3 [x,y,z] coordinates
    '''
    H = np.shape(dmap)[0]
    W = np.shape(dmap)[1]
    dmap = dmap.reshape((H, W))
    k_mat[0, :] = scale_x * k_mat[0, :]
    k_mat[1, :] = scale_y * k_mat[1, :]
    k_inv = np.linalg.inv(k_mat)
    num_pts = np.sum(np.greater(dmap, min_disp))
    pts = np.ones((3, num_pts))
    ctr = 0
    for y in range(H):
        for x in range(W):
            if (dmap[y, x] > min_disp) and (ctr < num_pts):
                pts[0, ctr] = x + 0.5
                pts[1, ctr] = y + 0.5
                pts[:, ctr] *= (1 / dmap[y, x])
                ctr += 1
    return np.transpose(np.matmul(k_inv, pts))


def points_to_cubes(points, edge_size=0.05):
    '''
    Converts an input point cloud to a set of cubes.
    
    Args:
        points: N X 3 array
        edge_size: cube edge size
    Returns:
        vs: vertices
        fs: faces
    '''
    v_counter = 0
    tot_points = points.shape[0]
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3).reshape(3, 12 * tot_points).transpose()
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for px in range(points.shape[0]):
        v_all[v_counter:v_counter + 8, :] *= edge_size
        v_all[v_counter:v_counter + 8, :] += points[px, :]
        v_counter += 8

    return v_all, f_all


def plot_cube(ax, corner_points):
    def cubify(points):
        # cube_definition_array = [
        #     np.array(list(item))
        #     for item in cube_definition
        # ]
        #
        # points = []
        # points += cube_definition_array
        # vectors = [
        #     cube_definition_array[1] - cube_definition_array[0],
        #     cube_definition_array[2] - cube_definition_array[0],
        #     cube_definition_array[3] - cube_definition_array[0]
        # ]
        #
        # points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        # points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        # points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        # points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]
        #
        # points = np.array(points)

        edges = [
            [points[0], points[3], points[2], points[1]],
            [points[1], points[6], points[5], points[2]],
            [points[7], points[6], points[5], points[4]],
            [points[0], points[7], points[4], points[3]],
            [points[0], points[1], points[6], points[7]],
            [points[3], points[2], points[5], points[4]]
        ]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
        faces.set_facecolor((0, 0, 1, 0.1))

        ax.add_collection3d(faces)
        # Plot the points themselves to force the scaling of the axes
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

    cubify(corner_points)
    return

def vis_detections(im, dets):
    """Visual debugging of detections."""
    for i in range(len(dets)):
        bbox = tuple(int(np.round(x)) for x in dets[i][0:4])
        im = im.copy()
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 0), 2)
        cv2.putText(im, '{}'.format(i), (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 0, 255), thickness=2)
    return im

def vis_relative_dirs(im, relative_dirs):
    """Visual debugging of detections."""
   
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(relative_dirs))
    for ix in range(len(relative_dirs)):
        direction, src_i, trj_i = relative_dirs[ix]
        src = tuple(int(np.round(x)) for x in direction[0])
        trj = tuple(int(np.round(x)) for x in direction[1])
        im = im.copy()
        color = tuple(int(255*x) for x in cm.jet(norm(ix))[0:3])
        cv2.arrowedLine(im, src, trj, color, 2, 8, 0, 0.1)
    return im


# def plot_cube_plotly(corner_points, obj_type='gt'):
    def cubify(points):

        if obj_type == 'gt':
            color = '#FFB6C1'
        else:
            color = '#B6C1FF'

        trace1 = go.Mesh3d(
        x=points[:, 0].tolist(),
        y=points[:,1].tolist(),
        z=points[:,2].tolist(),
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color=color,opacity=0.50, showlegend=False)
        trace2 = go.Scatter3d(
            x=points[:,0].tolist(),
            y=points[:,1].tolist(),
            z=points[:,2].tolist(),
            mode='markers', marker=dict(
                    size=1
                    ),showlegend=False
        )
        return [trace1, trace2]

    return cubify(corner_points)

import pdb
# import matplotlib.pyplot as plt
# if __name__ =="__main__":
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     cube_points = np.array([[0,0,0],
#                             [1,0,0],
#                             [1,1,0],
#                             [0,1,0],
#
#                             [0,0,1],
#                             [1,0,1],
#                             [1,1,1],
#                             [0,1,1]])
#     pdb.set_trace()
#     plot_cube(ax, cube_points)

if __name__ =="__main__":

    cube_points = np.array([[0,0,0],
                            [1,0,0],
                            [1,1,0],
                            [0,1,0],

                            [0,0,1],
                            [1,0,1],
                            [1,1,1],
                            [0,1,1]])
    data = plot_cube_plotly(cube_points)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                nticks=4, range=[-2, 2], ),
            yaxis=dict(
                nticks=4, range=[-2, 2], ),
            zaxis=dict(
                nticks=4, range=[-2, 2], ), ),
        width=700,
        margin=dict(
            r=20, l=10,
            b=10, t=10)
    )
    fig = go.Figure(data=data, layout=layout)
    pdb.set_trace()
    offline.plot(fig, show_link=False, auto_open=False)
    # py.image.save_as(fig, filename='test.png')
