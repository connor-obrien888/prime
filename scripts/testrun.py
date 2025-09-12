import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar, Timer
import argparse
import omegaconf


# Add the prime_torch file to the system path so we can import it
import sys
sys.path.append("/glade/u/home/cobrien/prime/prime_lib/primesw")
from data import SWDataset, SWDataModule
from prime_torch import crps, SWRegressor

def main(config):
    cfg = omegaconf.OmegaConf.load(
        config
    )

    datamodule = SWDataModule(
        target_features = cfg.data.target_features,
        input_features = cfg.data.input_features,
        position_features = cfg.data.position_features,
        cadence = cfg.data.cadence,
        window = cfg.data.window,
        stride = cfg.data.stride,
        interp_frac = cfg.data.interp_frac,
        trn_bounds = cfg.data.trn_bounds,
        val_bounds = cfg.data.val_bounds,
        tst_bounds = cfg.data.tst_bounds,
        batch_size = cfg.opt.batch_size,
        num_workers = cfg.opt.num_workers,
        datastore = cfg.data.datastore,
        key = cfg.data.key,
    )
    datamodule.setup()

    model = SWRegressor(
        optimizer = cfg.opt.optimizer,
        lr = cfg.opt.lr,
        lr_scheduler = cfg.opt.lr_scheduler,
        weight_decay = cfg.opt.weight_decay,
        in_dim = len(cfg.data.in_dim),
        tar_dim = len(cfg.data.tar_dim),
        pos_dim = len(cfg.data.pos_dim),
        decoder_type = cfg.model.decoder_type,
        encoder_type = cfg.model.encoder_type,
        decoder_hidden_layers = cfg.model.decoder_hidden_layers,
        encoder_hidden_dim = cfg.model.encoder_hidden_dim,
        encoder_num_layers=cfg.model.encoder_num_layers,
        p_drop = cfg.model.p_drop,
        pos_encoding_size=cfg.model.pos_encoding_size,
        loss=cfg.opt.loss
    )

    # TODO: implement tensorboard logger
    tb_logger = pl.loggers.TensorBoardLogger(
        cfg.experiments.trainer.tensorboard_path
    )

    trainer = pl.Trainer(
        accelerator=cfg.experiments.trainer.accelerator,
        max_epochs=cfg.experiments.trainer.max_epochs,
        callbacks = [Timer(), RichProgressBar()],
        # logger = tb_logger,
        # precision='16-true', #Lower the precision to not blow up memory
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single training run of PRIME.")
    parser.add_argument(
        "--config",
        type=str,
        default="~/prime/prime_lib/configs/testing_config.yaml",
        help="Path to config file defining training run.",
    )
    args = parser.parse_args()
    main(args.config)