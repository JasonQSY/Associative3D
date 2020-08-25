# Associative3D: Volumetric Reconstruction from Sparse Views

Code release for our paper

```
Associative3D: Volumetric Reconstruction from Sparse Views
Shengyi Qian*, Linyi Jin*, David Fouhey
European Conference on Computer Vision (ECCV) 2020
```

![teaser](static/teaser.png)

Please check the [project page](https://jasonqsy.github.io/Associative3D/) for more details and consider citing our paper if it is helpful:

```
@inproceedings{Qian20, 
    author = {Qian, Shengyi and Jin, Linyi and Fouhey, David},
    title = {Associative3D: Volumetric Reconstruction from Sparse Views},
    booktitle = {ECCV}, 
    year = {2020} 
}
```

## Setup

We use Ubuntu 18.04 and python 3.7 to train and test our approach. We use the same python environment for all of our experiments, except blender comes with its own python environment (which we cannot avoid), and detectron2 has its own dependencies.

Dependencies which require special attentions:
- blender 2.81. https://www.blender.org/
- CUDA 10.1, including `nvcc`.
- PyTorch 1.2.0. Make sure it supports GPU and `torch.version.cuda` matches your system CUDA version. We need compile our own C++ modules.

Install other python packages

```bash
pip install visdom dominate
pip install absl-py
pip install spherecluster
```

Compile C++ modules

```
# compile bbox_util.so
# pay attention to the actual name of .so file.
cd utils
python setup.py build_ext --inplace
cp relative3d/utils/bbox_utils.cpython-37m-x86_64-linux-gnu.so bbox_utils.so

# compile chamferdist
# this is a GPU implementation of Chamfer Distance
cd utils/chamferdist
python setup.py install
```

We use Blender 2.81 to visualize results. We install `imageio` to the python environment embedded with Blender. Note that the blender script is just for visualization and we've migrated it from 2.7 to 2.8 based on 3D-RelNet, which is a major API change.

## Dataset

To setup the preprocess SUNCG dataset for training and evaluation, we follow steps by [3D-RelNet](https://github.com/nileshkulkarni/relative3d#training-and-evaluating). In addition, please download our pre-computed Faster-RCNN proposals [here](https://www.dropbox.com/s/lvghv2kzfo3riz8/faster_rcnn_proposals.tar.gz?dl=0) and put it under the dataset. For example, `/x/syqian/suncg` is our SUNCG directory, while `/x/syqian/suncg/faster_rcnn_proposals` stores all pre-computed detection proposals.


## Object Branch

First we train the object branch. Please download our pre-computed files [here](https://www.dropbox.com/s/qh51k0fbwpcdbw3/object_branch_cachedir.tar.gz?dl=0) including translation and rotation centroids, data splits and pre-trained models.

**Training from scratch:** The first step is to train single-view 3D prediction network using ground truth bbox. This is exactly 3D-RelNet without finetuned on edgebox detections and NYUv2.

```bash
python -m object_branch.experiments.suncg.box3d --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_single --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=0 --num_epochs=8 --suncg_dir /x/syqian/suncg --pred_relative=True --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals  --auto_rel_opt=5  --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True
```

Before finetuning the object embedding, you may want to run the evaluation provided by 3D-RelNet to make sure the performance of your model matched the number reported by the 3D-RelNet paper.

```bash
python -m object_branch.benchmark.suncg.box3d --num_train_epoch=7 --name=box3d_single --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=True --suncg_dir=/x/syqian/suncg --preload_stats=False --results_name=box3d_local --do_updates=True --save_predictions_to_disk=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True
```

After having a working model for single-view predictions, we can continue training the object embedding.

```bash
python -m object_branch.experiments.suncg.siamese --plot_scalars --batch_size=24 --name=exp22_10-4_wmse --use_context --pred_voxels=False --classify_rot --save_epoch_freq=1 --shape_loss_wt=10 --n_data_workers=0 --num_epochs=30 --suncg_dir=/x/syqian/suncg --pred_relative=True --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True --ft_pretrain_epoch=8 --ft_pretrain_name=box3d_single --split_file=object_branch/cachedir/associative3d_train_set.txt --save_latest_freq=500 --id_loss_type=wmse --nz_id=64
```

## Camera Branch

Then we train the camera branch. Again, please download our pre-computed files [here](https://www.dropbox.com/s/kttlke25bv06fo8/camera_branch_cached_dir.tar.gz?dl=0) including translation and rotation centroids, data splits and pre-trained models. We put it at `camera_branch/cached_dir`.

**Training from scratch:**

```bash
python train.py --dataset_dir /x/syqian/suncg/renderings_ldr --exp_name debug --split_dir cached_dir/split
```

Before generating final predictions for relative camera pose, you may want to evaluate it first.

```bash
python evaluate.py --exp_name debug --phase validation
```

To generate final predictions, run

```bash
python inference.py --split-file cached_dir/split/test_set.txt --dataset-dir /x/syqian/suncg/renderings_ldr
```

and put the pkl file (e.g. `rel_poses_v3_test.pkl`) under `object_branch/cachedir`.

## Stitching Stage

We continue using the codebase of the object branch in our stitching stage. The predicted relative pose is saved in `rel_poses_v3_test.pkl`. To run the stitching stage, go to `object_branch/stitching`.

Run stitching for debug.
- set `--num_process=0` so that `pdb.set_trace()` works.
- set `--save_objs` so that we can visually check.

```bash
python stitch.py --suncg_dir=/x/syqian/suncg --output_dir=/x/syqian/output --num_process=0 --num_gpus=1 --split_file=../cachedir/associative3d_test_set.txt --rel_pose_file=../cachedir/rel_poses_v3_test.pkl --save_objs
```

Run stitching for visualization.
- `--num_process=32` and `--num_gpus=8` mean we're running 32 inference processes on 8 gpus. Therefore, each gpu has 4 processes.
- set `--save_objs` so that we can visually check.

```bash
python stitch.py --suncg_dir=/x/syqian/suncg --output_dir=/x/syqian/output --num_process=32 --num_gpus=8 --split_file=../cachedir/associative3d_test_set.txt --rel_pose_file=../cachedir/rel_poses_v3_test.pkl --save_objs
```


Run stitching for full-scene evaluation.
- disable `--save_objs` speeds up the stitching significantly since generating obj files is time-consuming.
- GPU is required to run inference of the object branch and compute chamfer distance.

```bash
python stitch.py --suncg_dir=/x/syqian/suncg --output_dir=/x/syqian/output --num_process=32 --num_gpus=8 --split_file=../cachedir/associative3d_test_set.txt --rel_pose_file=../cachedir/rel_poses_v3_test.pkl
```

Run stitching for abalations which need ground truth bbox.
- option `--gtbbox` is available so that Faster-RCNN proposals will be replaced with ground truth bbox.

```bash
python stitch.py --suncg_dir=/x/syqian/suncg --output_dir=/x/syqian/output --num_process=32 --num_gpus=8 --split_file=../cachedir/associative3d_test_set.txt --rel_pose_file=../cachedir/rel_poses_v3_test.pkl --gtbbox
```


## Evaluation

Full-scene evaluation

```bash
python eval_detection.py --result_dir=/x/syqian/output --split_file=../object_branch/cachedir/associative3d_test_set.txt
```

evaluation of object correspondence

```bash
python eval_correspondence.py --result_dir=/x/syqian/output --split_file=../object_branch/cachedir/associative3d_test_set.txt
```

evaluation of relative camera pose

```bash
python evaluate_camera_pose.py --result_dir=/x/syqian/output --split_file=../object_branch/cachedir/associative3d_test_set.txt
```

## Acknowledgment

We reuse the codebase of [factored3d](https://github.com/shubhtuls/factored3d) and [relative3d](https://github.com/nileshkulkarni/relative3d) for our object branch.
The computation of chamfer distance is taken from [chamferdist](https://github.com/krrish94/chamferdist).
We are really grateful to all these researchers who have made their code available.