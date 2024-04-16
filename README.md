<div align="center">   
  
# StreamingFlow: Streaming Occupancy Forecasting with Asynchronous Multi-modal Data Streams via Neural Ordinary Differential Equation
</div>

This repo introduces StreamingFlow (CVPR2024 poster(hightlight)).

## Demo videos
Occupancy forecasting on nuScenes dataset (IoU/VPQ = 53.9/52.7 when evaluated for 4 frames with interval 0.5s).
[https://github.com/synsin0/StreamingFlow/assets/37300008/ee225603-8434-4825-b912-7f3fc7095c85]

Occupancy forecasting on Lyft dataset (IoU/VPQ = 56.9/55.9 when evaluated for 10 frames with interval 0.2s).
[https://github.com/synsin0/StreamingFlow/assets/37300008/54232c6c-4ae2-456a-9381-4df5c5624712]

Streaming forecasting: foreseeing the unseen future to 8s (IoU/VPQ = 32.5/29.8 when evaluated for 16 frames with interval 0.5s).
[https://github.com/synsin0/StreamingFlow/assets/37300008/4fd86c70-b7e5-421e-b297-153cb4a8e477]

Streaming forecasting: temporally dense prediction at 20Hz (IoU/VPQ = 45.3/42.7 when evaluated for 40 frames with interval 0.05s).
[https://github.com/synsin0/StreamingFlow/assets/37300008/2f67398f-8819-4422-9a50-4f87d0e2c9b3]

## Framework
![teaser](sources/streamingflow_framework.png)

## Abstract

Predicting the future occupancy states of the surrounding environment is a vital task for autonomous driving. However, current best-performing single-modality methods or multi-modality fusion perception methods are only able to predict uniform snapshots of future occupancy states and require strictly synchronized sensory data for sensor fusion. We propose a novel framework, StreamingFlow, to lift these strong limitations. StreamingFlow is a novel BEV occupancy predictor that ingests asynchronous multi-sensor data streams for fusion and performs streaming forecasting of the future occupancy map at any future timestamps. By integrating neural ordinary differential equations (N-ODE) into recurrent neural networks, StreamingFlow learns derivatives of BEV features over temporal horizons, updates the implicit sensor's BEV features as part of the fusion process, and propagates BEV states to the desired future time point. It shows good zero-shot generalization ability of prediction, reflected in the interpolation of the observed prediction time horizon and the reasonable inference of the unseen farther future period. Extensive experiments on two large-scale datasets,nuScenes and Lyft L5,  demonstrate that StreamingFlow significantly outperforms previous vision-based, LiDAR-based methods, and shows superior performance compared to state-of-the-art fusion-based methods. 

 

## License

All assets and code are under the [Apache 2.0 license](https://github.com/synsin0/StreamingFlow/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:
```
to be updated
```
<!-- ```
@misc{shi2023fusionmotion,
      title={FusionMotion: Multi-Sensor Asynchronous Fusion for Continuous Occupancy Prediction via Neural-ODE}, 
      author={Yining Shi and Kun Jiang and Ke Wang and Jiusi Li and Yunlong Wang and Diange Yang},
      year={2023},
      eprint={2302.09585},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

## Acknowledgements
Thanks to prior excellent open source projects:

- [MotionNet](https://github.com/pxiangwu/MotionNet)
- [FIERY](https://github.com/wayveai/fiery)
- [ST-P3](https://github.com/OpenPerceptionX/ST-P3)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
- [StretchBEV](https://github.com/kaanakan/stretchbev)