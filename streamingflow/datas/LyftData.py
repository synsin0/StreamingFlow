import os
from PIL import Image

import numpy as np
import cv2
import torch
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box
from functools import reduce
from streamingflow.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)
from streamingflow.utils.instance import convert_instance_mask_to_center_and_offset_label
from streamingflow.utils.data_classes import LidarPointCloud
from streamingflow.utils.data_utils import voxelize_occupy, calc_displace_vector, point_in_hull_fast
import yaml

TRAIN_LYFT_INDICES = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16,
                      17, 18, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32,
                      33, 35, 36, 37, 39, 41, 43, 44, 45, 46, 47, 48, 49,
                      50, 51, 52, 53, 55, 56, 59, 60, 62, 63, 65, 68, 69,
                      70, 71, 72, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84,
                      86, 87, 88, 89, 93, 95, 97, 98, 99, 103, 104, 107, 108,
                      109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 121, 122, 124,
                      127, 128, 130, 131, 132, 134, 135, 136, 137, 138, 139, 143, 144,
                      146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159,
                      161, 162, 165, 166, 167, 171, 172, 173, 174, 175, 176, 177, 178,
                      179]

VAL_LYFT_INDICES = [0, 2, 4, 13, 22, 25, 26, 34, 38, 40, 42, 54, 57,
                    58, 61, 64, 66, 67, 77, 80, 85, 90, 91, 92, 94, 96,
                    100, 101, 102, 105, 106, 112, 120, 123, 125, 126, 129, 133, 140,
                    141, 142, 145, 155, 160, 163, 164, 168, 169, 170]

def range_projection(current_vertex, proj_H=64, proj_W=900, fov_up=3.0, fov_down=-25.0, max_range=50, min_range=2):
  """ Project a pointcloud into a spherical projection, range image.
    Args:
      current_vertex: raw point clouds
    Returns:
      proj_vertex: each pixel contains the corresponding point (x, y, z, depth)
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points

  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > min_range) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > min_range) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  proj_x_orig = np.copy(proj_x)
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  proj_y_orig = np.copy(proj_y)
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]
  
  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                           dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, depth]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_vertex

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class FuturePredictionDatasetLyft(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.is_train = is_train
        self.cfg = cfg

        self.is_lyft = isinstance(nusc, LyftDataset)

        if self.is_lyft:
            self.dataroot = self.nusc.data_path
        else:
            self.dataroot = self.nusc.dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future_frames = cfg.N_FUTURE_FRAMES
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()
        self.voxel_size = cfg.VOXEL.VOXEL_SIZE
        self.area_extents = np.array(cfg.VOXEL.AREA_EXTENTS)
        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        rv_config_filename = 'streamingflow/configs/nuscenes/data_preparing.yaml'
        if yaml.__version__ >= '5.1':
            self.rv_config = yaml.load(open(rv_config_filename), Loader=yaml.FullLoader)
        else:
            self.rv_config = yaml.load(open(rv_config_filename))

    def get_scenes(self):

        if self.is_lyft:
            scenes = [row['name'] for row in self.nusc.scene]

            # Split in train/val
            indices = TRAIN_LYFT_INDICES if self.is_train else VAL_LYFT_INDICES
            scenes = [scenes[i] for i in indices]
        else:
            # filter by scene split
            split = {'v1.0-trainval': {True: 'train', False: 'val'},
                     'v1.0-mini': {True: 'mini_train', False: 'mini_val'},}[
                self.nusc.version
            ][self.is_train]

            scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_lidar_range_data(self, sample_rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """


        if self.cfg.GEN.GEN_RANGE:
            # points = np.zeros((5, 0))
            points = np.zeros((5, 0))
            V = 35000 * nsweeps
            # Get reference pose and timestamp.
            ref_sd_token = sample_rec['data']['LIDAR_TOP']
            ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
            ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
            ref_time = 1e-6 * ref_sd_rec['timestamp']

            # Homogeneous transformation matrix from global to _current_ ego car frame.
            car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                                inverse=True)

            # Aggregate current and previous sweeps.
            sample_data_token = sample_rec['data']['LIDAR_TOP']
            current_sd_rec = self.nusc.get('sample_data', sample_data_token)
            sample_sd_rec = current_sd_rec
            for _ in range(nsweeps):
                # Load up the pointcloud and remove points close to the sensor.
                current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
                current_pc.remove_close(min_distance)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                    Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
                times = time_lag * np.ones((1, current_pc.nbr_points()))

                new_points = np.concatenate((current_pc.points, times), 0)
                points = np.concatenate((points, new_points), 1)

                if current_sd_rec['prev'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])
                    
                    
                    
            if points.shape[1] > V:
                    # # print('lidar_data.shape[0]', lidar_data.shape[0])
                    # np.random.shuffle(lidar_data)
                points = points[:,:V]
            elif points.shape[0] < V:
                points = np.pad(points,[(0,0),(0, V-points.shape[1])],mode='constant')  
                # Abort if there are no previous sweeps.
            range_view = range_projection(points.transpose(1,0),  
                                        self.rv_config['range_image']['height'], self.rv_config['range_image']['width'],
                                        self.rv_config['range_image']['fov_up'], self.rv_config['range_image']['fov_down'],
                                        self.rv_config['range_image']['max_range'], self.rv_config['range_image']['min_range'])

            rv_file_name = os.path.split(sample_sd_rec['filename'])[-1] 
            # os.makedirs(os.path.join(self.dataroot,'range_nusc',sample_sd_rec['channel']), exist_ok=True)                           
            # np.save(os.path.join(self.dataroot,'range_nusc', sample_sd_rec['channel'], rv_file_name+'.npy'),range_view)
        else:
            sample_data_token = sample_rec['data']['LIDAR_TOP']
            sample_sd_rec = self.nusc.get('sample_data', sample_data_token)
            range_view = np.load(os.path.join(self.dataroot,'range_nusc',sample_sd_rec['channel'], os.path.split(sample_sd_rec['filename'])[-1]+'.npy'))                           
        return torch.from_numpy(range_view).unsqueeze(0).to(torch.float32)
    
    def get_depth_from_lidar(self, lidar_sample, cam_sample):
  
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample['token'], cam_sample['token'])
        cam_file_name = os.path.split(cam_sample['filename'])[-1]
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(np.int)
        tmp_cam[points[1, :], points[0,:]] = coloring

        return tmp_cam

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )


            # Get Depth
            # Depth data should under the dataroot path 
            # if self.cfg.LIFT.GT_DEPTH:
            #     if self.cfg.GEN.GEN_DEPTH:
            #         depth = self.get_depth_from_lidar(lidar_sample,camera_sample)   # online without preprocessing
            #     else:
            #         import time
            #         t1 = time.time()
            #         depth = np.load(os.path.join(self.dataroot,'depth_nusc',camera_sample['channel'], os.path.split(camera_sample['filename'])[-1]+'.npy'))
            #         t2 = time.time()
            #         print('get one depth image:', t2-t1)
                
            #     depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            #     depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
            #     depth = depth.squeeze()
            #     crop = self.augmentation_parameters['crop']
            #     depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
            #     depth = torch.round(depth)
            #     depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        # if len(depths) > 0:
        #     depths = torch.cat(depths, dim=1)

        return images, intrinsics, extrinsics
    def get_lidar_data(self, sample_rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """
        # points = np.zeros((5, 0))
        points = np.zeros((5, 0))
        V = 35000 * nsweeps
        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])
                
                
                
        if points.shape[1] > V:
                # # print('lidar_data.shape[0]', lidar_data.shape[0])
                # np.random.shuffle(lidar_data)
            points = points[:,:V]
        elif points.shape[0] < V:
            points = np.pad(points,[(0,0),(0, V-points.shape[1])],mode='constant')  
            # Abort if there are no previous sweeps.

        return points
        # return torch.from_numpy(range_projection(points.transpose(1,0),  
        #                             self.rv_config['range_image']['height'], self.rv_config['range_image']['width'],
        #                             self.rv_config['range_image']['fov_up'], self.rv_config['range_image']['fov_down'],
        #                             self.rv_config['range_image']['max_range'], self.rv_config['range_image']['min_range'])).unsqueeze(0).to(torch.float32)
    
    
    def get_radar_data(self, rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """
        
        # points = np.zeros((5, 0))
        
        V = 700 * nsweeps
        points = np.zeros((19, 0))
        # Get reference pose and timestamp.
        ref_sd_token = rec['data']['RADAR_FRONT']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),inverse=True)

        # RadarPointCloud.disable_filters()
        RadarPointCloud.default_filters()

        # Aggregate current and previous sweeps.
        # from all radars 
        radar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
        for radar_name in radar_chan_list:
            sample_data_token = rec['data'][radar_name]
            current_sd_rec = self.nusc.get('sample_data', sample_data_token)
            for _ in range(nsweeps):
                # Load up the pointcloud and remove points close to the sensor.
                current_pc = RadarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
                current_pc.remove_close(min_distance)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                    Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
                times = time_lag * np.ones((1, current_pc.nbr_points()))

                new_points = np.concatenate((current_pc.points, times), 0)
                points = np.concatenate((points, new_points), 1)

    
              
              
                
                
                # print('time_lag', time_lag)
                # print('new_points', new_points.shape)

                # Abort if there are no previous sweeps.
                if current_sd_rec['prev'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])

        if points.shape[1] > V:
            points = points[:,:V]        
        elif points.shape[0] < V:
            points = np.pad(points,[(0,0),(0, V-points.shape[1])],mode='constant')
        

        radar_voxels = voxelize_occupy(points[:3,:].T, voxel_size=self.voxel_size, extents=self.area_extents)
        radar_voxels = radar_voxels.sum(-1)
        return torch.from_numpy(radar_voxels).unsqueeze(0).unsqueeze(0).to(torch.float32)


    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        # ref_cs_rec = self.nusc.get('calibrated_sensor', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        # yaw = Quaternion(ref_cs_rec['rotation']).yaw_pitch_roll[0]
        # rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_birds_eye_view_label(self, rec, instance_map):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)
            category_name = annotation['category_name']
            if not self.is_lyft:
                # NuScenes filter
                # if 'vehicle' not in annotation['category_name']:
                #     continue
                if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1:
                    continue
            else:
                # Lyft filter
                if annotation['category_name'] not in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
                    continue


            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]

            if not self.is_lyft:
                instance_attribute = int(annotation['visibility_token'])
            else:
                instance_attribute = 0

            poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
            cv2.fillPoly(instance, [poly_region], instance_id)
            
            cv2.fillPoly(segmentation, [poly_region], 1.0)
            
            # cv2.fillPoly(z_position, [poly_region], z)
            # cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

        return segmentation, instance, instance_map

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    # def get_label(self, rec, instance_map):
    #     segmentation_np, instance_np, z_position_np, instance_map, attribute_label_np = \
    #         self.get_birds_eye_view_label(rec, instance_map)
    #     segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
    #     instance = torch.from_numpy(instance_np).long().unsqueeze(0)
    #     z_position = torch.from_numpy(z_position_np).float().unsqueeze(0).unsqueeze(0)
    #     attribute_label = torch.from_numpy(attribute_label_np).long().unsqueeze(0).unsqueeze(0)

    #     return segmentation, instance, z_position, instance_map, attribute_label

    def get_label(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, instance_map = \
            self.get_birds_eye_view_label(rec, instance_map)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)

        return segmentation, instance, instance_map


    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def __len__(self):
        return len(self.indices)

    def get_temporal_voxels(self,index):
        if self.cfg.GEN.GEN_VOXELS:

            
            rec_curr_token = self.ixes[self.indices[index][self.receptive_field - 1]]
            curr_sample_data = self.nusc.get('sample_data', rec_curr_token['data']['LIDAR_TOP'])
            nsweeps_back = 5
            frame_skip = 1
            num_sweeps = nsweeps_back
            # Get the synchronized point clouds
            all_pc, all_times = LidarPointCloud.from_file_multisweep_bf_sample_data(self.nusc, curr_sample_data,
                                                                                    nsweeps_back=nsweeps_back,nsweeps_forward=0)
            # Store point cloud of each sweep
            pc = all_pc.points
            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)
           
            pc_list = []
            


            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                pc_list.append(_pc.T)
                

            selected_times = unique_times
            # Reorder the pc, and skip sample frames if wanted
            tmp_pc_list_1 = pc_list
            # selected_times = -selected_times[::-1] + 0.5 * (self.receptive_field - 1)
         
            tmp_pc_list_1 = tmp_pc_list_1[::-1]


            num_past_pcs = len(tmp_pc_list_1)

            # print('rec_field: ', self.receptive_field)
            # print('rec_samples: ', num_past_pcs)
            assert num_past_pcs == 5
            
            # latest_points = tmp_pc_list_1[0][:,:2].T
            # import cv2 as cv
            # os.makedirs('./output_lidar', exist_ok=True)
            # img = np.zeros((256, 256, 3), np.uint8)
            # point_size = 1
            # point_color = (0, 0, 255) # BGR
            # thickness = 4 # 可以为 0 、4、8
            # cv2.circle(img, latest_points.astype(int), point_size, point_color, thickness)
            # curr_sample_token = data['sample_token'][self.cfg.TIME_RECEPTIVE_FIELD - 1]       
            # cv2.imwrite(os.path.join('./output_lidar',curr_sample_token ) + '.png',img)

            # Voxelize the input point clouds, and compute the ground truth displacement vectors
            padded_voxel_points_list = list()  # This contains the compact representation of voxelization, as in the paper
        
            
            for i in range(num_past_pcs):
                vox = voxelize_occupy(pc_list[i], voxel_size=self.cfg.VOXEL.VOXEL_SIZE, extents=np.array(self.cfg.VOXEL.AREA_EXTENTS))
                padded_voxel_points_list.append(vox)

            # Compile the batch of voxels, so that they can be fed into the network
            padded_voxel_points = np.stack(padded_voxel_points_list, 0).astype(np.float32)
        
            # os.makedirs(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel']), exist_ok=True)                           
            # np.save(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel'], os.path.split(curr_sample_data['filename'])[-1]+'.npy'),padded_voxel_points)
        else:
            rec_curr_token = self.ixes[self.indices[index][self.receptive_field - 1]]
            curr_sample_data = self.nusc.get('sample_data', rec_curr_token['data']['LIDAR_TOP'])
            padded_voxel_points = np.load(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel'], os.path.split(curr_sample_data['filename'])[-1]+'.npy'))   

        padded_voxel_points = torch.from_numpy(padded_voxel_points)
        return padded_voxel_points, selected_times


    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices' , 'camera_timestamp' ,
                ]
        for key in keys:
            data[key] = []
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            data['radar_pointclouds']=[]
        if self.cfg.MODEL.MODALITY.USE_LIDAR:
            if self.cfg.MODEL.LIDAR.USE_RANGE:
                data['range_clouds']=[]    
            if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
                data['padded_voxel_points']=[]
                data['lidar_timestamp']=[]

        instance_map = {}
        # Loop over all the frames in the sequence.
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]
            data['camera_timestamp'].append(rec['timestamp'])
            if i < self.receptive_field:
                images, intrinsics, extrinsics = self.get_input_data(rec)
                data['image'].append(images)
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
            segmentation, instance, instance_map = self.get_label(rec, instance_map, in_pred)

            future_egomotion = self.get_future_egomotion(rec, index_t)
            # hd_map_feature = self.voxelize_hd_map(rec)
            
            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['future_egomotion'].append(future_egomotion)
            # data['hdmap'].append(hd_map_feature)
            data['indices'].append(index_t)

            if self.cfg.MODEL.MODALITY.USE_RADAR:
                radar_pointcloud = self.get_radar_data(rec,nsweeps=1,min_distance=2.2)
                data['radar_pointclouds'].append(radar_pointcloud)    
            if self.cfg.MODEL.LIDAR.USE_RANGE:
                lidar_range_cloud = self.get_lidar_range_data(rec,nsweeps=1,min_distance=2.2)
                data['range_clouds'].append(lidar_range_cloud)
            # if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
            #     gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
            #     data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
            #     data['command'] = command
            #     trajs = self.get_trajectory_sampling(rec)
            #     data['sample_trajectory'] = torch.from_numpy(trajs).float()
        

        data['camera_timestamp'] = np.array(data['camera_timestamp'])
        data['camera_timestamp'] = (data['camera_timestamp'] - data['camera_timestamp'][0]) / 1e6
        
        if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
            padded_voxel_points, lidar_voxel_times = self.get_temporal_voxels(index)
            data['lidar_timestamp']  = lidar_voxel_times
            data['padded_voxel_points'].append(padded_voxel_points)        
        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'segmentation', 'instance', 'future_egomotion']:
                data[key] = torch.cat(value, dim=0)

        if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
            data['padded_voxel_points'] = torch.cat(data['padded_voxel_points'], dim=0)
        if self.cfg.MODEL.LIDAR.USE_RANGE:
            data['range_clouds'] = torch.cat(data['range_clouds'], dim=0)   
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            data['radar_pointclouds'] = torch.cat(data['radar_pointclouds'], dim=0)       
        
        data['target_point'] = torch.tensor([0., 0.])
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow
        return data