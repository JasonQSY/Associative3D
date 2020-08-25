"""
python3.4 custom_render_results.py  --obj_files meshes/code_gt_6.obj meshes/code_gt_5.obj --hostname vader --r 2 --delta_theta 30 --out_dir ../../cachedir/visualization/blender/ --out_name_prefix test --add_objects_one_by_one 1 --sz_x 320 --sz_y 240
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

import argparse, pprint


def parse_args(str_arg):
  parser = argparse.ArgumentParser(description='render_script_savio')
  
  parser.add_argument('--hostname', type=str, default='vader')
  parser.add_argument('--out_dir', type=str, default=None)
  parser.add_argument('--out_name_prefix', type=str, default='a')
  
  parser.add_argument('--obj_files', type=str, nargs='+')
  parser.add_argument('--sz_x', type=int, default=320)
  parser.add_argument('--sz_y', type=int, default=240)

  parser.add_argument('--delta_theta', type=float, default=30.)
  parser.add_argument('--r', type=float, default=2.)
  parser.add_argument('--format', type=str, default='png')
  parser.add_argument('--add_objects_one_by_one', type=int, default=1)
 

  if len(str_arg) == 0:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args(str_arg)

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

  if write_png_jpg:
    import scipy.misc
  name = args.out_name_prefix
  
  camera_xyz = np.zeros((2,3))
  lookat_xyz = np.zeros((2,3))
  i = 0
  r = args.r
  for l in [2]:
    for t in [-args.delta_theta]:
      i = i+1
      t = np.deg2rad(t)
      camera_xyz[i,l] = r*np.sin(t)
      camera_xyz[i,1] = r*np.cos(t) - r
  lookat_xyz[:,1] = -r

  jpg_dir = os.path.join(args.out_dir)#, 'jpg')
  # mask_dir = os.path.join(args.out_dir, 'mask')

  re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='BLENDER_RENDER',
    threads=1, render_format=args.format)

  re.mkdir_if_missing(os.path.join(jpg_dir))
  
  if args.add_objects_one_by_one:
    J = range(len(args.obj_files))
  else:
    J = [len(args.obj_files)-1]

  for j in J:
    shape_files = args.obj_files[:(j+1)]
  
    ims, masks, _ = re._render(shape_files, re._get_lighting_param_png(), vps=None,
        camera_xyz=camera_xyz, lookat_xyz=lookat_xyz,
        tmp_file=tmp_file, exr_files=exr_files, 
        deform_fns=[deform_fn])
    for i in range(len(ims)):
      im_ = np.concatenate((ims[i], masks[i][:,:,np.newaxis].astype(np.uint8)), axis=2)
      output_path = os.path.join(
        jpg_dir, '{:s}_render_{:02d}_of_{:02d}_vp{:03d}.png'.format(
          name, j+1, len(args.obj_files), i))
      scipy.misc.imsave(output_path, im_)
