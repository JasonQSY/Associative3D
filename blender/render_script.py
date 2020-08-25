"""
python3.4 render_script_savio.py  --obj_file meshes/code_gt_6.obj --layout_file meshes/layout_gt_6.obj --hostname vader --r 2 --delta_theta 30
"""

import os, sys
file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.dirname(file_path))
sys.path.insert(0, os.path.join(os.path.dirname(file_path), 'bpy'))
import bpy
import numpy as np
from imp import reload 
import timer

import render_utils as ru
import render_engine as re
import imageio
import argparse, pprint


def parse_args(str_arg):
  parser = argparse.ArgumentParser(description='render_script_savio')
  
  parser.add_argument('--hostname', type=str, default='vader')
  parser.add_argument('--out_dir', type=str, default='../../cachedir/visualization/blender/')
  # parser.add_argument('--shapenet_dir', type=str, default='/global/scratch/saurabhg/shapenet/')
  
  parser.add_argument('--obj_file', type=str)
  parser.add_argument('--layout_file', type=str)
  parser.add_argument('--sz_x', type=int, default=320)
  parser.add_argument('--sz_y', type=int, default=240)

  parser.add_argument('--delta_theta', type=float, default=30.)
  parser.add_argument('--r', type=float, default=2.)
  parser.add_argument('--format', type=str, default='png')
 

  if len(str_arg) == 0:
    parser.print_help()
    sys.exit(1)

  #args = parser.parse_args(str_arg)
  args, _ = parser.parse_known_args()

  pprint.pprint(vars(args))
  return args

def deform_fn(vs):
  out = []
  for vs_ in vs:
    _ = vs_*1.
    # _[:,0] = vs_[:,2]
    # _[:,2] = -vs_[:,0]
    out.append(_)
  return out

if __name__ == '__main__':
  args = parse_args(sys.argv[1:])
  print(args)

  # re._prepare(640, 480, use_gpu=False, engine='BLENDER_RENDER', threads=1)
  tmp_file = os.path.join('/dev', 'shm', 'bpy-' + str(os.getpid()) + '.' + args.format)
  exr_files = None;

  write_png_jpg = True 
  vis = False
  write_exr = False 

  #if write_png_jpg:
  #  import scipy.misc
  name = os.path.splitext(os.path.basename(args.obj_file))[0]
  print(name)
  bird_view_flag = False
  novel_view_flag = True
  #if name.startswith('view'):
  #  bird_view_flag = False
  #else:
  #  novel_view_flag = True
  #bird_view_flag = True
  
  
  if False:
    view_params = np.zeros((args.num_azimuth,4))
    view_params[:,3] = 0.001
    view_params[:,0] = (np.arange(args.num_azimuth)*1./(args.num_azimuth-1.)-0.5)*30. + 90.
    view_params[:,1] = 0
    view_params[:,2] = 180.
    view_params = view_params[:6,:]
  
  #camera_xyz = np.zeros((5,3))
  #lookat_xyz = np.zeros((5,3))
  
  i = 0
  r = args.r
  #for l in [0, 2]:
  #  for t in [-args.delta_theta, args.delta_theta]:
  #    i = i+1
  #    t = np.deg2rad(t)
  #    camera_xyz[i,l] = r*np.sin(t)
  #    camera_xyz[i,1] = r*np.cos(t) - r
  #lookat_xyz[:,1] = -r
  
  # bird view
  if bird_view_flag:
    camera_xyz = np.zeros((5,3))
    lookat_xyz = np.zeros((5,3))
    lookat_xyz[:, 1] = -r
    camera_xyz[1, 2] = -4
    camera_xyz[2, 2] = -6
    camera_xyz[3, 2] = -8
    camera_xyz[4, 2] = -10
  elif novel_view_flag:
    camera_xyz = np.zeros((5,3))
    lookat_xyz = np.zeros((5,3))
    i = 0
    for l in [0, 2]:
      for t in [-args.delta_theta, args.delta_theta]:
        i = i+1
        t = np.deg2rad(t)
        camera_xyz[i,l] = r*np.sin(t)
        camera_xyz[i,1] = r*np.cos(t) - r
    lookat_xyz[:,1] = -r
  else:
    camera_xyz = np.zeros((1, 3))
    lookat_xyz = np.zeros((1, 3))
    lookat_xyz[:,1] = -r
  #camera_xyz[0, 2] = -4
  #camera_xyz[1, 2] = -10

  jpg_dir = os.path.join(args.out_dir)#, 'jpg')
  # mask_dir = os.path.join(args.out_dir, 'mask')

  re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='BLENDER_RENDER',
    threads=1, render_format=args.format)

  #shape_files = [os.path.join(args.obj_file), os.path.join(args.layout_file)]
  shape_files = [os.path.join(args.obj_file)]
  
  ims, masks, _ = re._render(shape_files, re._get_lighting_param_png(), vps=None,
      camera_xyz=camera_xyz, lookat_xyz=lookat_xyz,
      tmp_file=tmp_file, exr_files=exr_files, 
      deform_fns=[deform_fn])
  
  # write files here
  if write_png_jpg:
      re.mkdir_if_missing(os.path.join(jpg_dir))
      # re.mkdir_if_missing(os.path.join(mask_dir))
       
      for i in range(len(ims)):
          im_ = np.concatenate((ims[i], masks[i][:,:,np.newaxis].astype(np.uint8)), axis=2)
          output_path = os.path.join(jpg_dir, '{:s}_render_{:03d}.png'.format(name, i))
          #scipy.misc.imsave(output_path, im_)
          imageio.imwrite(output_path, im_)
          
          # output_path = os.path.join(mask_dir, '{:s}_render_{:03d}.jpg'.format(name, i))
          # scipy.misc.imsave(output_path, masks[i])
          
  if vis:
      fig, axes = vu.subplot(plt, (2,5))
      axes = axes.ravel()[::-1].tolist()
      for j in range(5):
          im = ims[j][:,:,:3]
          ax = axes.pop()
          ax.set_axis_off()
          ax.imshow(im.astype(np.uint8))

          im = masks[j]
          ax = axes.pop()
          ax.set_axis_off()
          ax.imshow(im.astype(np.uint8))
