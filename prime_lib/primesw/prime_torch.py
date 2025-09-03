import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from loguru  import logger

from models import LinearEncoder
    
class SWRegressor(pl.LigtningModule):
    def __init__(
            self,
            optimizer = "adam",
            lr = 1e-3,
            lr_scheduler = None,
            weight_decay = 0,
            recurrent_type = 'gru',
            decoder_type = 'linear',
            encoder_type = 'gru',
            decoder_hidden_layers = [128],
            encoder_hidden_dim = 128,
            encoder_num_layers = 1,
            p_drop = 0.1,
            #Might need a section here to indicate how to handle position
            pos_encoding_size = None,
            loss = 'mae',
            *args,
            **kwargs,
    ):
        super().__init(*args, **kwargs) # Pass bonus arguments to the LightningModule
        self.save_hyperparameters() #inherited method from LightningModule

        # Optimiser Parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler


        



    def configure_optimizers(self):
        match (self.optimizer):
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise NameError(f"Unknown optimizer {optimizer}")
        # Select LR scheduler
        scheduler_config = None
        match self.lr_scheduler:
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.max_epochs,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.factor, patience=self.patience,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Add this required parameter!
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "exp":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.gamma,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case _:
                raise ValueError(f"Unsupported scheduler: {self.lr_scheduler}")

        # Return config based on whether a scheduler is used
        if scheduler_config is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config
            }
        else:
            return optimizer