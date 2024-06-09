from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime
import os
import time
from streamingflow.datas.NuscenesData import FuturePredictionDataset
from streamingflow.trainer import TrainingModule
from streamingflow.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from streamingflow.utils.network import preprocess_batch, NormalizeInverse
from streamingflow.utils.instance import predict_instance_segmentation_and_trajectories
from streamingflow.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy
from streamingflow.datas.dataloaders import prepare_dataloaders

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('panop_visualize_imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, continuous=False, dataroot=None,  n_future_frames=4, frame_skip=4,  draw=False):

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=False)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg

    cfg.N_FUTURE_FRAMES = n_future_frames
    cfg.DATASET.FRAME_SKIP = frame_skip

    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.DATASET.VERSION = 'trainval'

    # cfg.DATASET.VERSION = 'mini'
    # cfg.N_WORKERS= 0

    if continuous:
        cfg.DATASET.USE_MULTISWEEP = True
    if dataroot:
        cfg.DATASET.DATAROOT = dataroot
        cfg.DATASET.MAP_FOLDER = dataroot

    _, valloader = prepare_dataloaders(cfg)
    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))

    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        camera_timestamps = batch['camera_timestamp']
        future_egomotion = batch['future_egomotion']
        target_timestamp = batch['target_timestamp']

        if not trainer.is_lyft:
            command = batch['command']
            trajs = batch['sample_trajectory']
            target_points = batch['target_point']
        range_clouds = None
        radar_pointclouds = None  
        padded_voxel_points = None
        lidar_timestamps = None
        points = None
        if trainer.cfg.MODEL.MODALITY.USE_RADAR:
            radar_pointclouds = batch['radar_pointclouds']
        if trainer.cfg.MODEL.MODALITY.USE_LIDAR:
            if trainer.cfg.MODEL.LIDAR.USE_RANGE:
                range_clouds = batch['range_clouds']    
            if trainer.cfg.MODEL.LIDAR.USE_STPN or trainer.cfg.MODEL.LIDAR.USE_BESTI: 
                padded_voxel_points = batch['padded_voxel_points']
                lidar_timestamps = batch['lidar_timestamp']
            else:
                points = batch['points']
                lidar_timestamps = batch['lidar_timestamp']

        B = len(image)
        labels = trainer.prepare_future_labels(batch)

        t0 = time.time()

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion ,padded_voxel_points,camera_timestamps, points, lidar_timestamps, target_timestamp,
            )
        t1 = time.time()


        n_present = model.receptive_field
        
        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])
        
        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        if cfg.INSTANCE_SEG.ENABLED:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False, make_consistent=True
            )
            metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                     labels['instance'][:, n_present - 1:])
            
            # import ipdb;ipdb.set_trace()

            
        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )
            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                metric_planning_val[i](final_traj[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])

        t2 = time.time()

        n_present_max = output['segmentation'].shape[1]


        if draw:
            # pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            #         output, compute_matched_centers=False, make_consistent=True
            #     )
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(   # do not use instance flow at post-processing
                    output, compute_matched_centers=False, make_consistent=True
                )
            for i in range(1, n_present_max):
                figure_numpy = plot_prediction(batch, labels, image, output, pred_consistent_instance_seg, i, index,save_path, cfg)
            
             
    results = {}

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    if cfg.INSTANCE_SEG.ENABLED:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')


def plot_prediction(batch, labels, image, output, consistent_instance_seg,index_t,frame,save_path,cfg):
    
    # Process predictions
    consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=False
    )
    segmentation = labels['segmentation'][:, index_t - 1].detach()
    # Plot future trajectories
    unique_ids = torch.unique(consistent_instance_seg[0, index_t]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(consistent_instance_seg[0, index_t].cpu().numpy(), instance_map)
    # trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    # for instance_id in unique_ids:
    #     path = matched_centers[instance_id]
    #     for t in range(len(path) - 1):
    #         color = instance_colours[instance_id].tolist()
           
    #         cv2.line(trajectory_img, tuple(path[t].astype(int)), tuple(path[t + 1].astype(int)), color, 4)

    # # Overlay arrows
    # temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    # mask = ~ np.all(trajectory_img == 0, axis=2)
    # vis_image[mask] = temp_img[mask]

    # Plot present RGB frames and predictions
    val_w = 2.99
    cameras = cfg.IMAGE.NAMES
    image_ratio = cfg.IMAGE.FINAL_DIM[0] / cfg.IMAGE.FINAL_DIM[1]
    val_h = val_w * image_ratio
    fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w, val_w)
    gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )
    if index_t<=3:
        visulize_img = image[0,index_t-1]
    else:
        visulize_img = image[0,2]
   
    for imgi, img in enumerate(visulize_img):
        ax = plt.subplot(gs[imgi // 3, imgi % 3])
        showimg = denormalise_img(img.cpu())
        if imgi > 2:
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

        plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                     xycoords='axes fraction', fontsize=14)
        plt.imshow(showimg)
        plt.axis('off')

    ax = plt.subplot(gs[:, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.imshow(make_contour(vis_image[::-1, ::-1]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)

    batch['camera_timestamp'] = torch.round(batch['camera_timestamp'] * 100)/100
    timestamps = batch['camera_timestamp'].cpu().numpy()
    display_time = timestamps[0][index_t-1]
    display_time
    if index_t<=3:
        plt.annotate("Perception at step %.2fs" % display_time, (0.02, 0.95), c='black', xycoords='axes fraction', fontsize=10)
        # plt.text(s="Perception step {}s".format(timestamps[0][index_t-1]))
        # plt.text(1, 0, s="Perception step {}s".format(timestamps[0][index_t-1]))
    else:
        plt.annotate("Prediction at step %.2fs" % display_time, (0.02, 0.95), c='black', xycoords='axes fraction', fontsize=10)
    plt.axis('off')

    # plt.subplot(gs[:, 4])
    # showing = torch.zeros((200, 200, 3)).numpy()
    # showing[:, :] = np.array([255 / 255, 255 / 255, 255 / 255])

    # # drivable
    # area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # # lane
    # area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    # semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    # segmentation = segmentation.cpu().numpy()
    # # semantic_index = semantic_seg > 0
    # semantic_index = segmentation[0,0] > 0
    # showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])
    # showing = np.flip(showing)

    # pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    # pedestrian_index = pedestrian_seg > 0
    # showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])
    # plt.annotate("GT %.2fs" % display_time, (0.02, 0.95), c='black', xycoords='axes fraction', fontsize=10)
    # plt.imshow(make_contour(showing))
    # plt.axis('off')


    plt.draw()
    figure_numpy = convert_figure_numpy(fig)
    plt.savefig(os.path.join(save_path,('%04d' % frame) + ('%04d.png' % index_t)))
    plt.close()
    return figure_numpy


if __name__ == '__main__':
    parser = ArgumentParser(description='StreamingFlow evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--future-frames', default=4, type=int)
    parser.add_argument('--frame-skip', default=4, type=int)

    args = parser.parse_args()

    eval(args.checkpoint, args.continuous, args.dataroot, args.future_frames, args.frame_skip)
