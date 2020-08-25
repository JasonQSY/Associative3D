# This script convert dataset from suncg format to coco format
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
import sys
sys.path.append('..')
import os
import scipy.io as sio
import pickle
from tqdm import tqdm
import cv2
import json
from absl import app, flags
import numpy as np

from utils import suncg_parse
cachedir = '.'
cachename = 'v3_detection'

# flags
flags.FLAGS(['demo'])
flags.DEFINE_string('dataset_dir', '/home/jinlinyi/workspace/p-voxels/script/v3_rpnet_split_relative3d', 'path that stores/saves png format files')
flags.DEFINE_string('suncg_dir', '/x/syqian/suncg', 'path to suncg folder')
flags.DEFINE_string('mode', 'suncg', 'suncg -> has all gts / nyu -> no gts')

opts = flags.FLAGS
dataset_dir = opts.dataset_dir
mode = opts.mode
suncg_dir = opts.suncg_dir


def get_bbox(house_name, view_id, meta_loader=None):
    # read bbox
    gt_bbox_path = os.path.join(suncg_dir, 'bboxes_node', house_name, 
                                view_id + '_bboxes.mat')
    gt_bbox = sio.loadmat(gt_bbox_path)
    assert(meta_loader is not None)
    house_data = suncg_parse.load_json(
        os.path.join(suncg_dir, 'house', house_name, 'house.json'))
    _, objects_bboxes, node_ids = suncg_parse.select_ids_multi(
        house_data, gt_bbox, meta_loader=meta_loader, min_pixels=500)
    return list(objects_bboxes)


def xyxy2xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2-x1, y2-y1]


def get_json(phase='test'):
    dataset_dir = opts.dataset_dir
    mode = opts.mode
    suncg_dir = opts.suncg_dir
    data = {}
    data["images"] = []
    data["annotations"] = []
    data["categories"] = []
    if phase == 'nyu':
        for i in range(80):
            data["categories"].append({"supercategory": "object", "id": i+1, "name":str(i)})
    else:
        data["categories"].append({"supercategory": "object", "id": 1, "name":"object"})


    with open(os.path.join(dataset_dir, phase+'_set.txt'), 'r') as f:
        lines = f.readlines()[3:]
    img_paths = []
    for line in lines:
        img_paths.append(line.split(' ')[0])
        img_paths.append(line.split(' ')[8])
    print(f"Cache file not found, loading from {phase} set, totally {len(img_paths)} imgs")
    if mode == 'suncg' or mode == 'multiview':
        meta_loader = suncg_parse.MetaLoader(
            os.path.join(suncg_dir, 'ModelCategoryMappingEdited.csv')
        )
        idx = 0
        for img_path in tqdm(img_paths):
            house_name, view_id = img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]
            bboxs = get_bbox(house_name, view_id, meta_loader)
            image_id = house_name+'_'+view_id
            for bbox in bboxs:
                x,y,w,h = xyxy2xywh(bbox)
                data['annotations'].append({'iscrowd': 0, 'image_id': image_id, 'bbox': [x,y,w,h], 'category_id': 1, 'id': idx, 'area': w*h})
                idx += 1
            data["images"].append({'file_name': os.path.join(house_name, view_id + '_mlt.png'), 'height': 480, 'width': 640, 'id': image_id})
        with open(os.path.join(dataset_dir, 'detectron_{}.json'.format(phase)), 'w') as outfile:
            json.dump(data, outfile)
    elif mode == 'nyu':
        idx = 0
        for img_path in tqdm(img_paths):
            house_name, view_id = img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]
            bboxs = [[1,1,2,2]] # placeholder
            image_id = house_name+'_'+view_id
            for bbox in bboxs:
                x,y,w,h = xyxy2xywh(bbox)
                data['annotations'].append({'iscrowd': 0, 'image_id': image_id, 'bbox': [x,y,w,h], 'category_id': 1, 'id': idx, 'area': w*h})
                idx += 1
            data["images"].append({'file_name': os.path.join(house_name, view_id + '.png'), 'height': 480, 'width': 640, 'id': image_id})
        with open(os.path.join(dataset_dir, 'detectron_{}.json'.format(phase)), 'w') as outfile:
            json.dump(data, outfile)
    else:
        raise NotImplementedError


def get_dicts(phase='test'):
    cache_f = os.path.join(cachedir, cachename + '_' + phase + '.pkl')
    if os.path.exists(cache_f):
        with open(cache_f, 'rb') as f:
            rtn = pickle.load(f)
        return rtn

    with open(os.path.join(dataset_dir, phase+'_set.txt'), 'r') as f:
        lines = f.readlines()[3:]
    img_paths = []
    for line in lines:
        img_paths.append(line.split(' ')[0])
        img_paths.append(line.split(' ')[8])
    print(f"Cache file not found, loading from {phase} set, totally {len(img_paths)} imgs")
    meta_loader = suncg_parse.MetaLoader(
        os.path.join(suncg_dir, 'ModelCategoryMappingEdited.csv')
    )
    rtn = []
    for img_path in tqdm(img_paths):
        house_name, view_id = img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]
        bboxs = get_bbox(house_name, view_id, meta_loader)
        annotations = []
        for bbox in bboxs:
            annot = {'bbox': list(bbox), 'category_id': 0, 'bbox_mode': BoxMode.XYXY_ABS, 'iscrowd': 0}
            annotations.append(annot)
        image_id = house_name+'_'+view_id
        rtn.append({'file_name':img_path, 'image_id': image_id, 'annotations': annotations, 
        "height":480, "width":640})
    try:
        with open(cache_f, 'wb') as f:
            pickle.dump(rtn, f)
    except:
        import pdb;pdb.set_trace()
        pass
    return rtn


def get_train_set():
    return get_dicts('train')


def get_test_set():
    return get_dicts('test')


def get_validation_set():
    return get_dicts('validation')


def main(_):
    dataset_dir = opts.dataset_dir
    mode = opts.mode
    suncg_dir = opts.suncg_dir
    if mode == 'nyu':
        get_json('nyu')
    elif mode == 'suncg':
        get_json('train')
        get_json('test')
        get_json('validation')
        print(f'train {len(get_train_set())}')
        print(f'test {len(get_test_set())}')
        print(f'val {len(get_validation_set())}')
    elif mode == 'multiview':
        get_json('multiview')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    app.run(main)