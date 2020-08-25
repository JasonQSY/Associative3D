## Train
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --num-gpus 4 --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_suncg.yaml
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 nice -n 10 python evaluate.py --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_suncg.yaml --eval-only OUTPUT_DIR /w/jinlinyi/faster_rcnn_models/evaluate_0044999 MODEL.WEIGHTS /w/jinlinyi/faster_rcnn_models/model_0044999.pth
```
