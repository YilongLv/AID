# AID
## ICCV 2023 Paper: [Anchor-Intermediate Detector: Decoupling and Coupling Bounding Boxes for Accurate Object Detection](https://arxiv.org/pdf/2310.05666)

## Abstract

Anchor-based detectors have been continuously developed for object detection. However, the individual anchor box makes it difficult to predict the boundary's offset accurately. Instead of taking each bounding box as a closed individual, we consider using multiple boxes together to get prediction boxes. To this end, this paper proposes the \textbf{Box Decouple-Couple(BDC) strategy} in the inference, which no longer discards the overlapping boxes, but decouples the corner points of these boxes. Then, according to each corner's score, we couple the corner points to select the most accurate corner pairs. To meet the BDC strategy, a simple but novel model is designed named the \textbf{Anchor-Intermediate Detector(AID)}, which contains two head networks, i.e., an anchor-based head and an anchor-free \textbf{Corner-aware head}. The corner-aware head is able to score the corners of each bounding box to facilitate the coupling between corner points. Extensive experiments on MS COCO show that the proposed anchor-intermediate detector respectively outperforms their baseline RetinaNet and GFL method by $\sim$2.4 and $\sim$1.2 AP on the MS COCO test-dev dataset without any bells and whistles.

<div align=center>
<img src="https://github.com/YilongLv/AID/blob/main/network.png"/>
</div>


## Install MMDetection and MS COCO2017
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmdet==2.15.1 and mmcv-full==1.3.17
## Add and Replace the codes
  - Add the configs/aid/. in our codes to your configs/. in mmdetectin's codes.
  - Add the mmdet/models/dense_heads/gfl_head_aid.py (& retina_head_aid.py) in our codes to your mmdet/models/dense_heads/. in mmdetectin's codes.
  - Replace the location of your 'MS-COCO' folder in  your configs/_base_/datasets/coco_detection.py in mmdetectin's codes.
## Train

```
#single GPU
python tools/train.py configs/aid/gfl_r50_fpn_1x_coco_aid.py

#multi GPU
bash tools/dist_train.sh configs/aid/gfl_r50_fpn_1x_coco_aid.py 8
```

## Test

```
#single GPU
python tools/test.py configs/aid/gfl_r50_fpn_1x_coco_aid.py $new_mmdet_ckpt --eval bbox

#multi GPU
bash tools/dist_test.sh configs/aid/gfl_r50_fpn_1x_coco_aid.py $new_mmdet_ckpt 8 --eval bbox
```
## Results

|    Model    |  Backbone  | Lr schd | Multi-scale Training | Baseline(mAP) | AID(mAP) |                                             config                                             |                                                     weight                                                      | code |
| :---------: |:----------:|:-------:|:--------------------:|:-------------:|:--------:|:----------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:| :--: |
|  RetinaNet  | ResNet-50  |   1x    |          No          |     36.5      |   37.6   |     [config](https://github.com/YilongLv/AID/configs/aid/retinanet_r50_fpn_1x_coco_aid.py)     |                            [baidu]()                             |  |
|  RetinaNet  | ResNet-101 |   1x    |          No          |     38.5      |   39.8   | [config](https://github.com/YilongLv/AID/configs/aid/retinanet_r50_fpn_mstrain_2x_coco_aid.py) |                            [baidu]()                             |      |
|     GFL     | ResNet-50  |   1x    |          No          |     40.2      |   41.0   |    [config](https://github.com/YilongLv/AID/configs/aid/gfl_r50_fpn_1x_coco_aid.py)            |                            [baidu]()                             |      |
|     GFL     | ResNet-50  |   2x    |         yes          |     42.9      |   43.5   |    [config](https://github.com/YilongLv/AID/configs/aid/gfl_r50_fpn_mstrain_2x_coco_aid.py)    |                            [baidu]()                             |      |
|     GFL     | ResNet-101 |   2x    |         yes          |     44.7      |   45.6   |   [config](https://github.com/YilongLv/AID/configs/aid/gfl_r101_fpn_mstrain_2x_coco_aid.py)    |                            [baidu]()                             |      |

\[1\] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
\[2\] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \

## Citation
```
@inproceedings{lv2023anchor,
  title={Anchor-intermediate detector: Decoupling and coupling bounding boxes for accurate object detection},
  author={Lv, Yilong and Li, Min and He, Yujie and Li, Shaopeng and He, Zhuzhen and Yang, Aitao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6275--6284},
  year={2023}
}
```


## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).
