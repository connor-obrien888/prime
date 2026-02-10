import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
import torchmetrics
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from loguru  import logger
import warnings

from models import LinearDecoder, RecurrentEncoder, TSPassthroughEncoder
    
class SWRegressor(pl.LightningModule):
    def __init__(
            self,
            optimizer = "adam",
            lr = 1e-3,
            lr_scheduler = None,
            patience=3,
            factor=0.5,
            weight_decay = 0,
            total_iters = 40,
            in_dim = 14,
            tar_dim = 1,
            pos_dim = 3,
            in_norm = None,
            tar_norm = None,
            pos_norm = None,
            window = 1,
            stride = 1,
            interp_frac = 1,
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
        self.total_iters = total_iters # Used for certain LR schedulers
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.factor = factor

        # Model Parameters
        self.in_dim = in_dim
        self.tar_dim = tar_dim
        self.pos_dim = pos_dim
        self.in_norm = in_norm
        self.tar_norm = tar_norm
        self.pos_norm = pos_norm
        self.window = window
        self.stride = stride # Only included so that it's saved as a hyperparameter
        self.interp_frac = interp_frac # Same as above
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.decoder_hidden_layers = decoder_hidden_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.p_drop = p_drop
        self.pos_encoding_size = pos_encoding_size

        # Loss parameters
        self.loss = loss
        # if self.loss == 'mae': #NOTE: I don't think we need any additional scalars logged for MAE-trained models? Besides the loss, which is handled later.
            # self.trn_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
            # self.val_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
            # self.tst_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
        if self.loss == 'crps':
            self.trn_mae = ProbabilisticMeanAbsoluteError()
            self.val_mae = ProbabilisticMeanAbsoluteError()
            self.tst_mae = ProbabilisticMeanAbsoluteError()

        # Initialize the encoder
        match self.encoder_type:
            case "rnn":
                self.encoder = RecurrentEncoder(
                    in_dim = self.in_dim,
                    encoding_size = self.encoder_hidden_dim,
                    num_layers = self.encoder_num_layers,
                    p_drop = self.p_drop,
                )
                decoder_in_dim = self.encoder_hidden_dim
            case "linear":
                self.encoder = TSPassthroughEncoder(
                    in_dim = self.in_dim * self.window,
                )
                decoder_in_dim = self.in_dim * self.window
            case _:
                raise ValueError(f"Invalid encoder type {self.encoder_type}")
        
        # Initialize the decoder
        match self.decoder_type:
            case "linear":
                self.decoder = LinearDecoder(
                    in_dim = decoder_in_dim,
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
                    in_dim = decoder_in_dim,
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
                self.loss_fn = torch.nn.L1Loss()
            case "crps":
                self.loss_fn = lambda outputs, targets: crps(
                    outputs,
                    targets,
                ).mean()
            case _:
                raise ValueError(f"Invalid loss type {self.loss}")
            
        # Define things we keep around for validation purposes, not passed to the model
        self.val_predictions = []
        self.val_targets = []
        self.val_times = []

        # Define an example input pair for generating the model graph
        self.example_input_array = (torch.rand(50, self.window, self.in_dim, device = self.device), torch.rand(50, self.pos_dim, device = self.device)) # (x, position)
    
    def forward(self, x, position):
        out, h = self.encoder.forward(x)
        y_hat = self.decoder.forward(out, position)
        return y_hat

    def predict(self, timeseries, position): # User-facing prediction step that scales data up and down automatically (human unit in, human unit out
        in_scaled = timeseries.loc[:, self.in_norm.keys()].copy() # Get just the keys used for prediction
        for feature in self.in_norm.keys(): # Scale each input feature DOWN
            in_scaled[feature] = (in_scaled[feature] - self.in_norm[feature][0])/self.in_norm[feature][1]

        # Turn in_scaled into a numpy array of the correct shape
        in_arr = np.zeros((len(position) - self.window, self.window, len(self.in_norm.keys())))
        for i, idx in enumerate(in_scaled.index):
            if i < self.window:
                continue
            in_arr[i - self.window, :, :] = in_scaled.loc[(idx - self.window - self.stride):(idx - self.stride - 1), :]

        pos_scaled = position.iloc[self.window:].loc[:, self.pos_norm.keys()].copy() # Get just the position elements
        for feature in self.pos_norm.keys(): # Scale each position DOWN
            pos_scaled[feature] = (pos_scaled[feature] - self.pos_norm[feature][0])/self.pos_norm[feature][1]
        
        # Tensor-ify the inputs from pandas dataframes
        in_tensor = torch.from_numpy(in_arr.astype(np.float32)).to(self.device)
        pos_tensor = torch.from_numpy(pos_scaled.to_numpy().astype(np.float32)).to(self.device)

        y_hat = self.forward(in_tensor, pos_tensor) # Run an actual forward pass
        y_hat = y_hat.detach().cpu().numpy()

        # Try to initialize the return dataframe
        try:
            tar_scaled = pd.DataFrame(timeseries['Epoch'].iloc[self.window:] + pd.Timedelta(seconds = self.stride * 100), columns = ['Epoch'])
        except KeyError: # If there is no 'Epoch' in the supplied dataframe
            warnings.warn('timeseries DataFrame does not have Epoch key, no time data will be returned')
            tar_scaled = pd.DataFrame([], index = timeseries.index[self.window:])
        
        if y_hat.shape[1] == 2*len(self.tar_norm.keys()): # If the output is means + stdevs
            for i, feature in enumerate(self.tar_norm.keys()):
                tar_scaled[feature] = (y_hat[:, i*2] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]
                tar_scaled[feature + '_std'] = ((y_hat[:, i*2] + y_hat[:, i*2 + 1]) * self.tar_norm[feature][1]) + self.tar_norm[feature][0] - tar_scaled[feature]
        else:
            for i, feature in enumerate(self.tar_norm.keys()):
                tar_scaled[feature] = (y_hat[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]
        
        return tar_scaled
    
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
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.trn_mae.update(y_hat, target)

        self.log(
            'Loss/train',
            loss.mean(),
            on_step=True,     # Log every step
            on_epoch=True,    # Log at end of epoch
            prog_bar=True,    # Show in progress bar
            logger=True,
            sync_dist=True
        )
        # Log current learning rate from optimizer
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('Opt/lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        val_loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.val_mae.update(y_hat, target)
        self.log('Loss/val', val_loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Store the batches so we can make a 2D joint distribution at epoch end
        self.val_predictions.append(y_hat.cpu())
        self.val_targets.append(target.cpu())
        self.val_times.append(times)

        # Plot the graph on the first validation epoch
        # self.log_graph()
        
        return val_loss

    def test_step(self, batch, batch_idx):
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        test_loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.tst_mae.update(y_hat, target)
        self.log('Loss/test', test_loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            "predictions": y_hat,
            "targets": target,
            "test_loss": test_loss,
            "timestamps": times,
        }
    
    def on_validation_epoch_end(self):
        # Compute and log all accumulated metrics
        if self.loss == 'crps':
            self.log('MAE/val', self.val_mae.compute().mean(), on_epoch = True, prog_bar = True, logger = True, sync_dist = True)
            # Clear all the metrics
            self.val_mae.reset()

        val_preds = torch.cat(self.val_predictions, dim = 0).numpy()
        targets = torch.cat(self.val_targets, dim = 0).numpy()
        if val_preds.shape[-1] == (self.tar_dim * 2): # Are we using one that outputs a mean and a standard deviation?
            predictions = val_preds[:, ::2]
            # logger.info(f"Plotting JD of probabilistic predictions of size {predictions.shape}")
        else:
            predictions = val_preds
            # logger.info(f"Plotting JD of deterministic predictions of size {predictions.shape}")
        fig, ax = plt.subplots(nrows = 1, ncols = self.tar_dim, figsize = (6 * self.tar_dim, 6))
        nbins = 50
        if self.tar_dim == 1: # In the case of a single target parameter, the Axes object will not be subscriptable
            ax = [ax] # Increase the dimensions of ax so that the indexing below still works
        for i, feature in enumerate(self.tar_norm.keys()):
            im = ax[i].hexbin(
                (targets[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0],
                (predictions[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0],
                gridsize = nbins,
                norm = LogNorm(1e0, 1e3),
                cmap = 'inferno', # TODO: make a fun new colormap
            )
            # ax[i].set_aspect("equal")
            ax[i].set_xlabel(f"Target {i}")
            ax[i].set_ylabel(f"Predcted {i}")
            ax[i].set_title(f"{feature}")
            lims = [
                np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
                np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
            ]
            ax[i].plot(
                lims,
                lims,
                color="k",
                linestyle = "--",
            )
            ax[i].set_aspect('equal')
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)
        self.logger.experiment.add_figure(f"JD/val_epoch{self.current_epoch}", fig)

        # TODO: Plot a holdout event

        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_times.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        if self.loss == 'crps':
            self.log('MAE/test', self.tst_mae.compute(), on_epoch=True, logger=True, sync_dist=True)

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
            case "cosine_warm":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.total_iters,
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
                    'monitor': 'Loss/val',  # Add this required parameter!
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1, end_factor=self.factor, total_iters=self.total_iters
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "const":
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=self.factor, total_iters=self.total_iters
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            # case "exp":
            #     scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #         optimizer, gamma=self.gamma,
            #     )
            #     scheduler_config = {
            #         'scheduler': scheduler,
            #         'interval': 'epoch',
            #         'frequency': 1
            #     }
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
        outputs = outputs.view(1, outputs.shape[0])
    if targets.dim() < 2:
        targets = targets.view(1, targets.shape[0])
    # This function uses the 1st, 3rd, 5th... neurons in the last layer as the means of the output 
    # Gaussian and the 2nd, 4th, 6th... neurons as the variance of the output Gaussians for each
    # target parameter. See http://www.dl.begellhouse.com/journals/52034eb04b657aea,3ec0b84376cff3d2,1801e97431c5911b.html
    # section 2 (equations 2 and 3) for more info. 
    ep = torch.abs(targets - outputs[:, ::2])
    loss = outputs[:, 1::2] * ((ep/outputs[:, 1::2]) * torch.erf((ep/(np.sqrt(2)*outputs[:, 1::2])))
                                + np.sqrt(2/np.pi) * torch.exp(-ep**2 / (2*outputs[:, 1::2]**2))
                                - 1/np.sqrt(np.pi))
    return loss

class GaussianContinuousRankedProbabilityScore(torchmetrics.Metric):
    # Like torchmetrics.regression.crps.ContinuousRankedProbabilityScore but takes the mean
    # and variance of a Gaussian instead of an ensemble of predictions as its input.
    # From https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    is_differentiable = True # Is the metric differentiable? Yes, the CRPS is differentiable.
    higher_is_better = False # Is a higher metric better (e.g. accuracy)? No, CRPS is like MAE where lower is better.
    full_state_update = False # Does .update() need to know the global metric state? No, each score is independent.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default = torch.tensor(0), dist_reduce_fx='mean')
    def update(self, preds, target):
        self.score = crps(preds, target)
    def compute(self):
        return self.score

class ProbabilisticMeanAbsoluteError(torchmetrics.Metric):
    # A version of MAE used as a metric for dual-output models trained with the CRPS.
    # Splits out the means of the distributions and uses them to calculate the MAE.
    # From https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    is_differentiable = True # Is the metric differentiable? Yes, the MAE is differentiable.
    higher_is_better = False # Is a higher metric better (e.g. accuracy)? No.
    full_state_update = False # Does .update() need to know the global metric state? No, each score is independent.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default = torch.tensor(0), dist_reduce_fx='mean')
    def update(self, preds, target):
        self.score = torch.nn.functional.l1_loss(preds[:, ::2], target)
    def compute(self):
        return self.score