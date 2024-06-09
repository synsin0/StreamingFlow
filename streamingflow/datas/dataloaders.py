import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from streamingflow.datas.NuscenesData import FuturePredictionDataset
from lyft_dataset_sdk.lyftdataset import LyftDataset
from streamingflow.datas.LyftData import FuturePredictionDatasetLyft

import os

def prepare_dataloaders(cfg, return_dataset=False):
    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=True)
        traindata = FuturePredictionDataset(nusc, 0, cfg)
        valdata = FuturePredictionDataset(nusc, 1, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'nuscenesmultisweep':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=True)
        traindata = FuturePredictionDatasetMultiSweep(nusc, 0, cfg)
        valdata = FuturePredictionDatasetMultiSweep(nusc, 1, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'lyft':
        # train contains 22680 samples
        # we split in 16506 6174
        # dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
        dataroot = cfg.DATASET.DATAROOT
        nusc = LyftDataset(data_path=dataroot,
                           json_path=os.path.join(dataroot, 'train_data'),
                           verbose=True)
        traindata = FuturePredictionDatasetLyft(nusc, 1, cfg)
        valdata = FuturePredictionDatasetLyft(nusc, 0, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)



    else:
        raise NotImplementedError

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader