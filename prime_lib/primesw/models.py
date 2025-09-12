import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class LinearDecoder(nn.Module):
    def __init__(
            self,
            in_dim,
            tar_dim,
            pos_dim,
            pos_encoding_size=4,
            hidden_layers=[128],
            p_drop=0.1,
        ):
        super().__init__()
        self.tar_dim = tar_dim
        self.in_dim = in_dim # NOTE: this is the input size from the recurrent encoder
        self.pos_dim = pos_dim
        self.network = None
        self.hidden_layers = hidden_layers
        self.p_drop = p_drop
        self.pos_encoding_size = pos_encoding_size

        # Calculate total input dimension after RFF position encoding
        pos_encoding_dim = 2 * self.pos_dim * (self.pos_encoding_size + 1) # +1 is for zeroth order RFF term
        total_input_dim = self.in_dim + pos_encoding_dim

        self.network = nn.Sequential( #Make the starting layer
            nn.Linear(total_input_dim, self.hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
        )
        for i, layer in enumerate(self.hidden_layers): # Add other layers according to list hidden_layers
            if i == 0: # We already added the first layer
                continue
            next_layer = nn.Sequential( #Make the next layer
                nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]),
                nn.ReLU(),
                nn.Dropout(p=self.p_drop),
            )
            self.network.extend(next_layer)
        self.network.extend(nn.Sequential(nn.Linear(self.hidden_layers[-1], self.tar_dim))) # Add the last layer
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, position):
        batch_size = x.size(1) # Recurrent-like networks in torch return tensors of shape (layer, batch, encoding_dim)
        # Flatten input while preserving batch dimension
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        pos_encoded = rff(
            position,
            max_encoding=self.pos_encoding_size
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        combined = torch.cat([x, pos_encoded], dim=-1)
  
        return self.network(combined)
    
class RecurrentEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            encoding_size=128,
            num_layers=1,
            p_drop=0.1,
            bidirectional=False,
        ):
        super().__init__()
        self.in_dim = in_dim # NOTE: this is the input size of the timeseries inputs
        self.encoding_size = encoding_size
        self.p_drop = p_drop
        self.network = nn.RNN(in_dim, encoding_size, num_layers, batch_first = True, dropout = p_drop, bidirectional = bidirectional)

    def forward(self, x):
        return self.network(x)


def rff(position, max_encoding = 4, include_raw_coordinates=False): # Random Fourier Features for position encoding
    if position.ndim == 1: # If position is 1D, add the extra dimension
        position = position.unsqueeze(0)
    powers = 2.0 ** torch.arange(max_encoding + 1, device=position.device, dtype=position.dtype) # Vector of powers of two up to max_encoding
    scaled_pos = position.unsqueeze(-1) * powers # NOTE: position should still be normed prior to this step. 'scaled' refers to the powers of two
    scaled_pos = scaled_pos.view(position.size(0), -1)
    cos_vec = torch.cos(scaled_pos) #RFF Cosine terms
    sin_vec = torch.sin(scaled_pos) #RFF Sine terms
    concat = torch.cat([cos_vec, sin_vec], dim=-1) # Add all RFF terms together
    if include_raw_coordinates:
        concat = torch.cat([concat, position])
    return concat