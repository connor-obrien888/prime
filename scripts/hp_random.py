import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar, Timer, LearningRateFinder
import argparse
import omegaconf


# Add the prime_torch file to the system path so we can import it
import sys
sys.path.append("/glade/u/home/cobrien/prime/prime_lib/primesw")
from data import SWDataset, SWDataModule
from prime_torch import crps, SWRegressor


def run_model(datamodule, config, hp_df, runname):
    '''
    Run one instance of the model according to the configuration (config) and the dictionary for what hyperparameters to use (hp_dict)
    '''
    model = SWRegressor(
        optimizer = config.opt.optimizer,
        lr = config.opt.lr,
        lr_scheduler = config.opt.lr_scheduler,
        patience = config.opt.patience,
        factor = config.opt.factor,
        weight_decay = config.opt.weight_decay,
        total_iters = config.opt.total_iters,
        in_dim = len(config.data.input_features),
        tar_dim = len(config.data.target_features),
        pos_dim = len(config.data.position_features),
        tar_norm = datamodule.target_normalizations,
        window = config.data.window,
        stride = config.data.stride,
        interp_frac = config.data.interp_frac,
        decoder_type = config.model.decoder_type,
        encoder_type = config.model.encoder_type,
        decoder_hidden_layers = [
            config.model.decoder_hidden_layers_1[int(hp_df.loc[0, 'decoder_hidden_layers_1'])],
            config.model.decoder_hidden_layers_2[int(hp_df.loc[0, 'decoder_hidden_layers_2'])],
            config.model.decoder_hidden_layers_3[int(hp_df.loc[0, 'decoder_hidden_layers_3'])],
            config.model.decoder_hidden_layers_4[int(hp_df.loc[0, 'decoder_hidden_layers_4'])],
        ],
        encoder_hidden_dim = config.model.encoder_hidden_dim[int(hp_df.loc[0, 'encoder_hidden_dim'])],
        encoder_num_layers=config.model.encoder_num_layers[int(hp_df.loc[0, 'encoder_num_layers'])],
        p_drop = config.model.p_drop[int(hp_df.loc[0, 'p_drop'])],
        pos_encoding_size= config.model.pos_encoding_size[int(hp_df.loc[0, 'pos_encoding_size'])],
        loss=config.opt.loss,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir = config.experiments.trainer.tensorboard_path,
        name = runname,
        log_graph = True,
    )

    trainer = pl.Trainer(
        accelerator=config.experiments.trainer.accelerator,
        max_epochs=config.experiments.trainer.max_epochs,
        callbacks = [
            Timer(), 
            RichProgressBar(),
            # LearningRateFinder(),
            # ModelCheckpoint(),
            ],
        logger = logger,
        # precision='16-true', #Lower the precision to not blow up memory
    )
    trainer.fit(model=model, datamodule=datamodule)
    return model

def get_and_log_unique_hp(config, logfile):
    log = pd.read_csv(logfile, index_col = 0)
    in_log = True
    while in_log:
        in_log = False
        hp_df = pd.DataFrame(columns = ['decoder_hidden_layers_1', 'decoder_hidden_layers_2', 'decoder_hidden_layers_3', 'decoder_hidden_layers_4', 'encoder_hidden_dim', 'encoder_num_layers', 'p_drop', 'pos_encoding_size'], index = [0])
        hp_df['decoder_hidden_layers_1'] = np.random.randint(0, len(config.model.decoder_hidden_layers_1))
        hp_df['decoder_hidden_layers_2'] = np.random.randint(0, len(config.model.decoder_hidden_layers_2))
        hp_df['decoder_hidden_layers_3'] = np.random.randint(0, len(config.model.decoder_hidden_layers_3))
        hp_df['decoder_hidden_layers_4'] = np.random.randint(0, len(config.model.decoder_hidden_layers_4))
        hp_df['encoder_hidden_dim'] = np.random.randint(0, len(config.model.encoder_hidden_dim))
        hp_df['encoder_num_layers'] = np.random.randint(0, len(config.model.encoder_num_layers))
        hp_df['p_drop'] = np.random.randint(0, len(config.model.p_drop))
        hp_df['pos_encoding_size'] = np.random.randint(0, len(config.model.pos_encoding_size))
        for idx in log.index:
            if (log.loc[idx, :] == hp_df.loc[0, :]).all():
                in_log = True # Continue the loop to generate a new HP set
    log = pd.concat([log, hp_df], ignore_index = True)
    log.to_csv(logfile)
    return hp_df


def main(config, runname, logfile, N):
    torch.set_float32_matmul_precision('medium')
    cfg = omegaconf.OmegaConf.load(
        config
    )

    datamodule = SWDataModule(
        target_features = cfg.data.target_features,
        input_features = cfg.data.input_features,
        position_features = cfg.data.position_features,
        interp_flags = cfg.data.interp_flags,
        region = cfg.data.region,
        cadence = cfg.data.cadence,
        interpolate = cfg.data.interpolate,
        window = cfg.data.window,
        stride = cfg.data.stride,
        interp_frac = cfg.data.interp_frac,
        trn_bounds = cfg.data.trn_bounds,
        val_bounds = cfg.data.val_bounds,
        tst_bounds = cfg.data.tst_bounds,
        batch_size = cfg.opt.batch_size,
        num_workers = cfg.opt.num_workers,
        datastore = cfg.data.datastore,
        in_key = cfg.data.in_key,
        tar_key = cfg.data.tar_key,
    )

    for i in range(N):
        hp_df = get_and_log_unique_hp(cfg, logfile)
        run_model(datamodule, cfg, hp_df, runname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single training run of PRIME.")
    parser.add_argument(
        "--config",
        type=str,
        default="/glade/u/home/cobrien/prime/prime_lib/configs/hpsearch_config.yaml",
        help="Path to config file defining hyperparameter search space.",
    )
    parser.add_argument(
        "--runname",
        type=str,
        default="graphtest",
        help="Name of this model run.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="/glade/u/home/cobrien/prime/prime_lib/configs/hplog.csv",
        help="Path to log file containing hyperparameter configurations that have already been tested.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=5,
        help="Number of models to train in the run.",
    )
    args = parser.parse_args()
    main(args.config, args.runname, args.logfile, args.N)