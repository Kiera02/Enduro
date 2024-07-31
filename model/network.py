import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.nn import MultiheadAttention
import math
from torchsummary import summary

class ImgPosEnc(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = torch.ones(x.size()[:3], dtype=torch.bool, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # not exactly the same as concat two WordPosEnc
        # WordPosEnc: sin(0), cos(0), sin(2), cos(2)
        # ImagePosEnc: sin(0), cos(1), sin(2), cos(3)
        dim_t = torch.arange(self.half_d_model, dtype=torch.float, device=x.device)
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x

class EncoderAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    
    def forward(self, x):
        return self.encoder(x)


class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, learning_rate, checkpoint_file):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.checkpoint_file = checkpoint_file

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 256, 3, 1)

        # Attention mechanism
        # Position encoding
        self.pos_enc = ImgPosEnc(d_model=256, temperature=10000.0, normalize=True)

        self.attention = EncoderAttention(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, num_encoder_layers=1)

        # Fully connected layers
        flattened_shape = self.calculate_flattened_shape(self.input_shape)
        self.fc1 = nn.Linear(flattened_shape, 512)
        self.fc2 = nn.Linear(512, output_shape)

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)

        # Device setup
        self.device = self.get_device()
        self.to(self.device)

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        
        device = torch.device(device_name)
        logging.info(f'Using device: {device}')
        return device

    def calculate_flattened_shape(self, input_shape):
      x = torch.zeros(1, *input_shape)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      return int(np.prod(x.size()))

    def save_checkpoint(self):
        logging.info('Saving checkpoint')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info('Loading checkpoint')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))

    def to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    def forward(self, inputs):
        # Convolutional layers
        inputs = inputs.to(self.device)
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))

        # Apply attention mechanism
        x = self.pos_enc(x.permute(0, 2, 3, 1))

        x = self.attention(x.flatten(1,2))

        # Fully connected layers
        x = f.relu(self.fc1(x.flatten(1,2)))
        x = self.fc2(x)
        
        # # without attention
        # # Flatten
        # x = x.view(x.size()[0], -1)
        # # Linear layers
        # x = f.relu(self.fc1(x))
        return x

    def backward(self, target, value):
        loss = self.loss(target, value).to(self.device)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    model = DeepQNetwork(input_shape=(4, 84, 84), output_shape=9, learning_rate=0.00025, checkpoint_file='checkpoint.pth')
    # model.load_checkpoint()
    # model.save_checkpoint()
    model.forward(torch.rand(2, 4, 84, 84))
    pass