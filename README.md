<div align="center">   
  
# StreamingFlow

StreamingFlow: Streaming Occupancy Forecasting with Asynchronous Multi-modal Data Streams via Neural Ordinary Differential Equation
</div>

This repo introduces StreamingFlow (CVPR2024 poster(hightlight)).

## Demo videos
Occupancy forecasting on nuScenes dataset

https://github.com/synsin0/StreamingFlow/assets/37300008/ee225603-8434-4825-b912-7f3fc7095c85

Occupancy forecasting on Lyft dataset 

https://github.com/synsin0/StreamingFlow/assets/37300008/54232c6c-4ae2-456a-9381-4df5c5624712

Streaming forecasting: foreseeing the future to 8s 

https://github.com/synsin0/StreamingFlow/assets/37300008/d67c3eff-822d-43d8-946f-f7bde8c4a693

Streaming forecasting: predicting at given interval 0.05s/0.10s/0.25s

https://github.com/synsin0/StreamingFlow/assets/37300008/5754bfc3-6649-40fa-a3b7-5647aaba7b3d

https://github.com/synsin0/StreamingFlow/assets/37300008/8f1ddc37-e5f8-492b-b553-8fb37e7e26e8

https://github.com/synsin0/StreamingFlow/assets/37300008/c051c049-083f-453b-845e-b609c1b55ae0


## Future (Ongoing) works
We implement StreamingFlow on Vidar codebase and generates streaming prediction on self-supervised 4d occupancy forecasting task with future point clouds as proxy. It is still in an early stage. We provide demo videos of current process.

Streaming forecasting with interval 0.5s: 

https://github.com/synsin0/StreamingFlow/assets/37300008/a1dbe140-c33b-4800-b433-70f100e5bf6d

Streaming forecasting with interval 0.05s: 

https://github.com/synsin0/StreamingFlow/assets/37300008/3509d5bd-7b4c-44f5-a9e7-2b02e4f94775


## Framework
![teaser](sources/streamingflow_framework.png)


## Abstract（TL DR）

StreamingFlow is a streaming occupancy forecasting framework which can input multi-modal asynchronous data streams (possibly with different given frequency) as input, and outputs future instance prediction in a continuous manner. 

## Installation and data setup

We follow the [ST-P3](https://github.com/OpenDriveLab/ST-P3) setup and [bevfusion](https://github.com/mit-han-lab/bevfusion) setup for environoment. For data setup, simply organize nuscenes and lyft dataset in ./data/nuscenes and ./data/lyft.


## Models

| Settings        | Image | LiDAR | ODE Step | IoU | VPQ | config  | checkpoint |
| ------------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| past_1s, future_2s | Effi-B4-224x480-2Hz   | Spconv8x-0050-5Hz     | variable    | 53.7    | 50.7 | [config](streamingflow/configs/Prediction_LC_ODE_Variable.yml) | [ckpt](https://cloud.tsinghua.edu.cn/f/0da4c5bd409a4a7bb80b/?dl=1) |

Train command:  
```python
python train.py --config /path/to/config
```
    

Test command: 
```python
python evaluate.py --checkpoint /path/to/checkpoint
```

## Experiments

We use streamingflow with variable ode step config and checkpoint to conduct the following experiments. 

### Predicting the unseen future exps

| Settings  | 1s | 2s | 3s | 4s | 5s | 6s | 8s  | 
| ------------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Variable  | 56.5/54.4 | 53.7/50.7 |  50.4/47.2  |  47.2/44.1  |  44.1/41.1  |  40.7/38.0  |  34.4/32.6   |

Test command: 
```python
 python evaluate.py --checkpoint /path/to/checkpoint --future-frames N 
```
here, N is for N * 0.5s future seconds.



### Predicting at any future interval


| Settings   | 0.05s  | 0.1s | 0.25s | 0.5s | 0.6s |  
| ------------- | ------- | -------- | -------- | -------- | -------- |
| Variable |  48.2/45.2    |   49.5/46.4   |    51.5/48.5   |   53.6/49.6   |   53.4/49.8   |

Test command: 
```python
export PYTHONPATH=/project_root_dir/nuscenes-devkit/python-sdk:$PYTHONPATH
python evaluate_streaming.py --checkpoint /path/to/checkpoint --eval-interval N 
```

here, N is for N * 0.05s interval.

### Predicting with different data stream intervals


| Settings     |    0.15s  | 0.2s | 0.25s |  0.4s | 0.5s |
| ------------- | ------- | -------- | -------- | -------- | -------- |
| Variable    |   53.1/50.0   |   53.7/50.7   |   53.2/50.3   |   50.6/47.4 |  47.6/44.5 |

Test command: 
```python
python evaluate_datastream.py --checkpoint /path/to/checkpoint --frame-skip N 
```

here, N is for 20/N interval for lidar input stream interval.







## License

All assets and code are under the [Apache 2.0 license](https://github.com/synsin0/StreamingFlow/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:
```
to be updated
```
<!-- ```
@misc{shi2023StreamingFlow,
      title={StreamingFlow: Multi-Sensor Asynchronous Fusion for Continuous Occupancy Prediction via Neural-ODE}, 
      author={Yining Shi and Kun Jiang and Ke Wang and Jiusi Li and Yunlong Wang and Diange Yang},
      year={2023},
      eprint={2302.09585},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

## Acknowledgements
Thanks to prior excellent open source projects:
- [GRU_ODE_Bayes](https://github.com/edebrouwer/gru_ode_bayes)
- [MotionNet](https://github.com/pxiangwu/MotionNet)
- [FIERY](https://github.com/wayveai/fiery)
- [ST-P3](https://github.com/OpenDriveLab/ST-P3)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
- [StretchBEV](https://github.com/kaanakan/stretchbev)
- [bevfusion](https://github.com/mit-han-lab/bevfusion)
- [Vidar](https://github.com/OpenDriveLab/ViDAR)
