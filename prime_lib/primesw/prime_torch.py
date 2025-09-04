import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
import gc
import numpy as np
import pandas as pd
from loguru  import logger

from models import LinearDecoder, RecurrentEncoder
    
class SWRegressor(pl.LightningModule):
    def __init__(
            self,
            optimizer = "adam",
            lr = 1e-3,
            lr_scheduler = None,
            weight_decay = 0,
            in_dim = 14,
            tar_dim = 1,
            pos_dim = 3,
            decoder_type = 'linear',
            encoder_type = 'rnn',
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
        super().__init__(*args, **kwargs) # Pass bonus arguments to the LightningModule
        self.save_hyperparameters() #inherited method from LightningModule

        # Optimiser Parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        # Model Parameters
        self.in_dim = in_dim
        self.tar_dim = tar_dim
        self.pos_dim = pos_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.decoder_hidden_layers = decoder_hidden_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.p_drop = p_drop
        self.pos_encoding_size = pos_encoding_size

        # Loss parameters
        # self.trn_mae = 
        # self.tst_mae = 
        # self.val_mae = 
        self.loss = loss

        # Initialize the encoder
        match self.encoder_type:
            case "rnn":
                self.encoder = RecurrentEncoder(
                    in_dim = self.in_dim,
                    encoding_size = self.encoder_hidden_dim,
                    num_layers = self.encoder_num_layers,
                    p_drop = self.p_drop,
                )
            case _:
                raise ValueError(f"Invalid encoder type {self.encoder_type}")
        
        # Initialize the decoder
        match self.decoder_type:
            case "linear":
                self.decoder = LinearDecoder(
                    in_dim = self.encoder_hidden_dim * self.encoder_num_layers,
                    tar_dim = self.tar_dim,
                    pos_dim = self.pos_dim,
                    pos_encoding_size = self.pos_encoding_size,
                    hidden_layers = self.decoder_hidden_layers,
                    p_drop = self.p_drop,
                )
            case "prob_linear": 
                # This is a special case of linear that outputs two values for each target feature.
                # NOTE: Compatible with loss = 'crps' ONLY!
                self.decoder = LinearDecoder(
                    in_dim = self.encoder_hidden_dim * self.encoder_num_layers,
                    tar_dim = self.tar_dim * 2,
                    pos_dim = self.pos_dim,
                    pos_encoding_size = self.pos_encoding_size,
                    hidden_layers = self.decoder_hidden_layers,
                    p_drop = self.p_drop,
                )
            case _:
                raise ValueError(f"Invalid decoder type {self.decoder_type}")
        
        # Handle the loss type
        match self.loss:
            case "mae":
                self.loss_fn = lambda outputs, targets: torch.nn.functional.L1Loss(
                    outputs,
                    targets,
                )
            case "crps":
                self.loss_fn = lambda outputs, targets: crps(
                    outputs,
                    targets,
                )
            case _:
                raise ValueError(f"Invalid loss type {self.loss}")
    
    def forward(self, x, position):
        out, h = self.encoder.forward(x)
        y_hat = self.decoder.forward(h, position)
        return y_hat
    
    def predict_step(self, batch, batch_idx):
        timeseries, position, target, times = batch
        with torch.no_grad():
            y_hat = self(timeseries, position)
            h = self.encoder.forward(timeseries)
        return {
            'inputs': timeseries,
            'positions': position,
            'encodings': h,
            'predictions': y_hat,
            'targets': target,
            'timestamps': times,
        }

    def training_step(self, batch, batch_idx):
        timeseries, position, target, _ = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        loss = self.loss_fn(y_hat, target)

        # TODO: Figure out logging (probably tensorboard)
        # self.log(
        #     'train_loss',
        #     loss,
        #     on_step=True,     # Log every step
        #     on_epoch=True,    # Log at end of epoch
        #     prog_bar=True,    # Show in progress bar
        #     logger=True,
        #     sync_dist=True
        # )
        # Log current learning rate from optimizer
        # lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log('lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        timeseries, position, target, _ = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        val_loss = self.loss_fn(y_hat, target)

        # TODO: Figure out logging (probably tensorboard)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        test_loss = self.loss_fn(y_hat, target)

        # TODO: Figure out logging (probably tensorboard)
        
        return {
            "predictions": y_hat,
            "targets": target,
            "test_loss": test_loss,
            "timestamps": times,
        }
    
    def on_validation_epoch_end(self):
        # TODO: Compute and log all accumulated metrics

        # TODO: Figure out logging (probably tensorboard)

        # TODO: Clear all the metrics
        # for metric in [self.val_f1,
        #                self.val_precision,
        #                self.val_recall,
        #                self.val_acc,
        #                self.val_auroc,
        #                self.val_mcc,
        #                self.val_kappa]:
        #     metric.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # TODO: Figure out logging (probably tensorboard)
        logger.info(f"Test epoch end. Reminder to implement logging.")

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = pl.utilities.grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms)

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

def crps(outputs, targets):
    if ((outputs.size(-1)%2)!=0):
        raise ValueError(f"CRPS loss function requires even number of outputs from model.")
    if outputs.dim() < 2: #If passed 1D outputs/targets
        outputs = outputs.view(1, outputs.shape(0))
    if targets.dim() < 2:
        targets = targets.view(1, targets.shape(0))
    ep = torch.abs(targets - outputs[:, ::2])
    loss = outputs[:, 1::2] * ((ep/outputs[:, 1::2]) * torch.erf((ep/(np.sqrt(2)*outputs[:, 1::2]))) + np.sqrt(2/np.pi) * torch.exp(-ep**2 / (2*outputs[:, 1::2]**2)) - 1/np.sqrt(np.pi))
    return loss