import bpy
import numpy as np
import os
import render_utils as ru
from timer import Timer
import copy
# from imp import reload
import imageio
import bmesh

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# for s in list(set(bpy.data.objects.keys()) - set(old)):
#   bpy.data.objects[s].select = True
# bpy.ops.object.delete()

def _prepare(im_x, im_y, use_gpu=False, engine='BLENDER_RENDER',
  gpu_device='CUDA_0', threads=1, render_format='png'):
  
  # Rendering parameters
  blank_blend = os.path.join(os.path.dirname(__file__), 'blank.blend')
  bpy.ops.wm.open_mainfile(filepath=blank_blend)
  #engine = 'BLENDER_EEVEE'
  #bpy.context.scene.render.engine = engine

  bpy.context.scene.render.resolution_percentage = 50
  bpy.context.scene.render.resolution_x = im_x*(100/bpy.context.scene.render.resolution_percentage)
  bpy.context.scene.render.resolution_y = im_y*(100/bpy.context.scene.render.resolution_percentage)
  #bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
  bpy.context.scene.render.film_transparent = True
  if render_format == 'png':
    bpy.context.scene.render.image_settings.file_format = 'PNG'
  elif render_format == 'exr':
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
  bpy.context.scene.render.image_settings.use_zbuffer = True
  bpy.context.scene.render.threads_mode = 'FIXED'
  bpy.context.scene.render.threads = threads
  # bpy.context.scene.render.subsurface_scattering.use = True
  # bpy.context.scene.render.use_shadows = False
  # bpy.context.scene.render.use_raytrace = False

  if use_gpu:
    bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    bpy.context.user_preferences.system.compute_device = gpu_device
    # bpy.context.scene.cycles.device = 'GPU'

  ###### Camera settings ######
  camObj = bpy.data.objects['Camera']
  camObj.data.lens = 517.97 #same as the one in sunch_pbrs
  camObj.data.sensor_height = 480.0
  camObj.data.sensor_width = float(camObj.data.sensor_height)/im_y*im_x
  # camObj.data.type = 'ORTHO'
  # camObj.data.ortho_scale = 1
  # camObj.data.lens_unit = 'FOV'
  # camObj.data.angle = 0.2

  ###### Compositing node ######
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree
  links = tree.links
  for n in tree.nodes:
      tree.nodes.remove(n)
  rl = tree.nodes.new(type="CompositorNodeRLayers")
  composite = tree.nodes.new(type = "CompositorNodeComposite")
  composite.location = 200,0
  links.new(rl.outputs['Image'],composite.inputs['Image'])
  #links.new(rl.outputs['Z'],composite.inputs['Z'])

  # Remove default lights 
  if 'Lamp' in bpy.data.objects.keys():
      bpy.data.objects['Lamp'].data.energy = 0
      bpy.ops.object.select_all(action='DESELECT')
      if 'Lamp' in list(bpy.data.objects.keys()):
          bpy.data.objects['Lamp'].select_set(True) # remove default light
      bpy.ops.object.delete()
  
  # print(bpy.data.objects['Camera'])
  # print(bpy.data.objects.keys())

def _get_lighting_param_exr(typ='default'):
  if typ == 'default':
    g_syn_light_num_lowbound = 20
    g_syn_light_num_highbound = 20
    g_syn_light_dist_lowbound = 8
    g_syn_light_dist_highbound = 20
    g_syn_light_azimuth_degree_lowbound = 0
    g_syn_light_azimuth_degree_highbound = 360
    g_syn_light_elevation_degree_lowbound = -90
    g_syn_light_elevation_degree_highbound = 90
    g_syn_light_energy_mean = 20
    g_syn_light_energy_std = 10
    g_syn_light_environment_energy_lowbound = 2
    g_syn_light_environment_energy_highbound = 2.01
    out = locals()
    out.pop("typ", None)
    return out


def _get_lighting_param_png(typ='default'):
  if typ == 'default':
    g_syn_light_num_lowbound = 2
    g_syn_light_num_highbound = 8
    g_syn_light_dist_lowbound = 8
    g_syn_light_dist_highbound = 20
    g_syn_light_azimuth_degree_lowbound = 0
    g_syn_light_azimuth_degree_highbound = 360
    g_syn_light_elevation_degree_lowbound = -90
    g_syn_light_elevation_degree_highbound = 90
    g_syn_light_energy_mean = 2
    g_syn_light_energy_std = 0.01
    g_syn_light_environment_energy_lowbound = 0
    g_syn_light_environment_energy_highbound = 2.01
    out = locals()
    out.pop("typ", None)
    return out

def set_all_vertices(mesh_names, vs):
  # get all mesh vertices for all meshes in a np.array
  for m, vv in zip(mesh_names, vs):
    sub_mesh = bpy.data.objects[m]
    for v, v_new in zip(sub_mesh.data.vertices, vv):
      v.co[0] = v_new[0]
      v.co[1] = v_new[1]
      v.co[2] = v_new[2]
    sub_mesh.data.update()
'''
def set_all_vertices(mesh_names, vs):
  for m, vv in zip(mesh_names, vs):
    sub_mesh = bpy.data.objects[m]
    mesh = bpy.data.meshes.new("mesh")
    bm = bmesh.new()
    #                     0          1       2         3         4         5       6        7
    block=np.array([ [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]).astype(float)
    block*=0.4
    verts = np.empty(shape=(0,3))
    from tqdm import tqdm
    for i in tqdm(vv):
        #print((block+i).shape)
        verts=np.append(verts, (block+i),axis=0)

    
    # for i in tqdm(vv):
    #     #print((block+i).shape)
    #     verts.append(block+i)
        
    #print(verts)
    for v in tqdm(verts):
        bm.verts.new(v)
    bm.to_mesh(mesh)
    bm.verts.ensure_lookup_table()
    for i in tqdm(range(0,len(bm.verts),8)):
        bm.faces.new( [bm.verts[i+0], bm.verts[i+1],bm.verts[i+3], bm.verts[i+2]])
        bm.faces.new( [bm.verts[i+4], bm.verts[i+5],bm.verts[i+1], bm.verts[i+0]])
        bm.faces.new( [bm.verts[i+6], bm.verts[i+7],bm.verts[i+5], bm.verts[i+4]])
        bm.faces.new( [bm.verts[i+2], bm.verts[i+3],bm.verts[i+7], bm.verts[i+6]])
        bm.faces.new( [bm.verts[i+5], bm.verts[i+7],bm.verts[i+3], bm.verts[i+1]]) #top
        bm.faces.new( [bm.verts[i+0], bm.verts[i+2],bm.verts[i+6], bm.verts[i+4]]) #bottom
    
    if bpy.context.mode == 'EDIT_MESH':
        bmesh.update_edit_mesh(sub_mesh.data)
    else:
        bm.to_mesh(sub_mesh.data)
    sub_mesh.data.update()
    bm.free
'''

def get_all_vertices(mesh_names):
  # get all mesh vertices for all meshes in a np.array
  vs = []
  for m in mesh_names:
    sub_mesh = bpy.data.objects[m]
    vv = [np.array(v.co) for v in sub_mesh.data.vertices]
    vv = np.array(vv)
    vs.append(vv)
  return vs

def _generate_grid(low=-1., high=1., step=0.5):
    # Generate source grid
    a = np.arange(low, high+step, step)
    Y, X, Z = np.meshgrid(a, a, a);
    out = np.concatenate((X[...,np.newaxis], Y[...,np.newaxis], Z[...,np.newaxis]), axis=3)
    return out, a

def deform_model_interpn(vertex_list, rng=None, delta=None):
  import scipy.interpolate
  # Create the interpolant function
  output_coords, a = _generate_grid()
  
  if rng is not None:
      output_coords = output_coords + (rng.rand(*(output_coords.shape))-0.5)*step/factor
  elif delta is not None:
      output_coords = output_coords + delta
      
  vs_all = np.concatenate(vertex_list, axis=0)
  cnt = np.array([v.shape[0] for v in vertex_list])
  cnt = np.cumsum(cnt)[:-1]
  output_vertices_all = scipy.interpolate.interpn((a,a,a), 
    np.transpose(output_coords, axes=[0,1,2,3]), 2*vs_all[:,[0,1,2]])[:,[0,1,2]]
  output_vertices_all = output_vertices_all/2.
  out_vertex_list = np.array_split(output_vertices_all, cnt)
  return out_vertex_list

def deform_model_linear(vertex_list, rng=None, delta=None, step=0.5, factor=3.):
  import scipy.interpolate
  # Create the interpolant function
  input_coords = _generate_grid()
  output_coords = _generate_grid()
  if rng is not None:
    output_coords = output_coords + (rng.rand(*(output_coords.shape))-0.5)*step/factor
  elif delta is not None:
    output_coords = output_coords + delta
  f = scipy.interpolate.LinearNDInterpolator(input_coords.reshape((-1,3)), output_coords.reshape((-1,3)))
  
  vs_all = np.concatenate(vertex_list, axis=0) 
  cnt = np.array([v.shape[0] for v in vertex_list])
  cnt = np.cumsum(cnt)[:-1]
  
  output_vertices_all = f(vs_all*2.)/2.
  
  out_vertex_list = np.array_split(output_vertices_all, cnt)
  return out_vertex_list

def deform_model():
  import scipy.interpolate
  # from scipy.interpolate import RegularGridInterpolator as rgi
  # my_interpolating_function = rgi((x,y,z), V)
  # Vi = my_interpolating_function(np.array([xi,yi,zi]).T)
  # for all object meshes, strech them by some amount
  model_objs = set(bpy.data.objects.keys()) - set(['Camera'])
  for mesh_name in model_objs:
    sub_mesh = bpy.data.objects[mesh_name] 
    for v in sub_mesh.data.vertices:
      v.co[2] = v.co[2]*2.
    sub_mesh.data.update()

def _render(shape_files, lighting_param, vps=None, rng=None, tmp_file=None,
  deform_fns=[None], camera_xyz=None, lookat_xyz=None, exr_files=None):
  # delete all objects except the camera
  to_delete_objects = set(bpy.data.objects.keys())-set(['Camera'])
  bpy.ops.object.select_all(action='DESELECT')
  for s in list(to_delete_objects):
    bpy.data.objects[s].select_set(True)
  bpy.ops.object.delete()
  
  # load the current object we want to render
  for s in shape_files:
    bpy.ops.import_scene.obj(filepath=s)
  model_objs = set(bpy.data.objects.keys()) - set(['Camera'])

  if vps is not None:
    N = vps.shape[0]
  elif camera_xyz is not None:
    N = camera_xyz.shape[0]
  
  assert(type(deform_fns) == list)
  if len(deform_fns) == 1:
    if deform_fns[0] is not None:
      vs = get_all_vertices(model_objs)
      out_vs = deform_fns[0](vs)
      set_all_vertices(model_objs, out_vs)
    deform_fns = [None]*N
  else:
    vs = get_all_vertices(model_objs)
    vs = copy.deepcopy(vs)

  
  ims = []
  masks = []

  # get the camObj 
  camObj = bpy.data.objects['Camera']
  tt = {}
  tt['setup'] = Timer()
  tt['render'] = Timer()
  tt['read'] = Timer()

  for j in range(N):
      deform_fn = deform_fns[j]
      rng = np.random.RandomState(0)

      if deform_fn is not None:
        out_vs = deform_fn(vs)
        set_all_vertices(model_objs, out_vs)
      
      tt['setup'].tic()
      


      # set environment lighting
      bpy.context.scene.world.use_nodes = True
      bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.1
      bpy.context.scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

      # set point lights
      for i in range(rng.randint(lighting_param['g_syn_light_num_lowbound'], lighting_param['g_syn_light_num_highbound']+1)):
        light_azimuth_deg   = rng.uniform(lighting_param['g_syn_light_azimuth_degree_lowbound'], lighting_param['g_syn_light_azimuth_degree_highbound'])
        light_elevation_deg = rng.uniform(lighting_param['g_syn_light_elevation_degree_lowbound'], lighting_param['g_syn_light_elevation_degree_highbound'])
        light_dist          = rng.uniform(lighting_param['g_syn_light_dist_lowbound'], lighting_param['g_syn_light_dist_highbound'])
        
        lx, ly, lz          = ru.obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        obj = set(bpy.data.objects.keys())

        # point light
        bpy.ops.object.light_add(type='POINT', align='WORLD', location=(lx, ly, lz))
        new_lamp_name = 'Point' if i == 0 else 'Point.{:03d}'.format(i)
        bpy.data.objects[new_lamp_name].data.energy = 3000

        # sun light (deprecated)
        #bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0,0,0), rotation=(0,0,0))
        #new_lamp_name = 'Sun' if i == 0 else 'Sun.{:03d}'.format(i)
        #bpy.data.objects[new_lamp_name].data.energy = rng.normal(lighting_param['g_syn_light_energy_mean'], lighting_param['g_syn_light_energy_std'])

      # set camera location
      if vps is not None:
        vp = vps[j,:]
        azimuth_deg = vp[0]
        elevation_deg = vp[1]
        theta_deg = -1 * vp[2] # ** multiply by -1 to match pascal3d annotations **
        rho = vp[3]
        cx, cy, cz = ru.obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
        q1 = ru.camPosToQuaternion(cx, cy, cz)
        q2 = ru.camRotQuaternion(cx, cy, cz, theta_deg)
        q = ru.quaternionProduct(q2, q1);
      else:
        assert(camera_xyz is not None)
        cx, cy, cz = camera_xyz[j,:]
        lx, ly, lz = lookat_xyz[j,:]
        q1 = ru.camPosToQuaternion(cx-lx, cy-ly, cz-lz)
        q2 = ru.camRotQuaternion(cx-lx, cy-ly, cz-lz, 180.)
        q = ru.quaternionProduct(q2, q1);

      camObj.location[0] = cx; camObj.location[1] = cy; camObj.location[2] = cz;
      camObj.rotation_mode = 'QUATERNION'
      camObj.rotation_quaternion[0] = q[0]; camObj.rotation_quaternion[1] = q[1]; 
      camObj.rotation_quaternion[2] = q[2]; camObj.rotation_quaternion[3] = q[3];
      tt['setup'].toc()
      
      tt['render'].tic()
      if tmp_file is not None:
        bpy.context.scene.render.filepath = tmp_file
      else:
        bpy.context.scene.render.filepath = exr_files[j]
      print(bpy.context.scene.render.filepath)
      bpy.ops.render.render(write_still=True)
      tt['render'].toc()
      
      # get camera matrix and write to file
      K = ru.get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
      # print(K)

      # read and return the image
      if tmp_file is not None:
        tt['read'].tic()
        ext = os.path.splitext(tmp_file)[-1]
        if ext == '.exr':
          import exrUtils as eu
          _, rgb, alpha = eu.imread(tmp_file)
          ims.append(rgb.astype(np.uint8))
          masks.append(alpha)
        elif ext == '.png':
          #import scipy.misc
          #im = scipy.misc.imread(tmp_file)
          im = imageio.imread(tmp_file)
          ims.append(im[:,:,:3])
          masks.append(im[:,:,3])
          # ims.append(scipy.misc.imread(tmp_file))
          # import pdb; pdb.set_trace()
        os.remove(tmp_file)
        tt['read'].toc()
      

      # cleanup, clear all lights
      tt['setup'].tic()
      #bpy.ops.object.select_by_type(type='LAMP')
      bpy.ops.object.select_by_type(type='LIGHT')
      bpy.ops.object.delete(use_global=False)
      tt['setup'].toc()
  return ims, masks, tt
