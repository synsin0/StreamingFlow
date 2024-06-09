import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import time
import socket
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint

from streamingflow.config import get_parser, get_cfg
from streamingflow.datas.dataloaders import prepare_dataloaders
from streamingflow.trainer import TrainingModule


def get_latest_checkpoint(folder_path):
    import glob
    import re
    import os

    ckpt_files = glob.glob(os.path.join(folder_path, '*.ckpt'))

    pattern = re.compile(r'epoch=(\d+).*\.ckpt')

    max_epoch = -1
    max_file_path = None

    for file_path in ckpt_files:
        match = pattern.match(os.path.basename(file_path))
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_file_path = file_path

    if max_file_path:
        print(f"The path to the .ckpt file with the highest epoch number is: {max_file_path}")
    else:
        print("No .ckpt files with the naming convention epoch_{}* were found.")
    
    return max_file_path


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH, map_location='cpu'
        )['state_dict']
        state = model.state_dict()
        pretrained_model_weights = {k: v for k, v in pretrained_model_weights.items() if k in state and 'decoder' not in k}
        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, cfg.TAG
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=-1,
        save_last=False,
        period=1,
        mode='max'
    )

    latest_ckpt = get_latest_checkpoint(save_dir)

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=False),
        profiler='simple',
        callbacks=[checkpoint_callback],
        resume_from_checkpoint = latest_ckpt

    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
