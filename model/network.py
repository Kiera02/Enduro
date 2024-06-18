import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.nn import MultiheadAttention


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
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # Attention mechanism
        self.attention = MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

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
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))

        # Reshape for attention
        x = x.permute(0, 2, 3, 1)  
        x = x.view(-1, x.size(1) * x.size(2), x.size(3))

        # Apply attention mechanism
        x, _ = self.attention(x, x, x)

        # Flatten
        x = x.view(x.size(0), -1)

         # Fully connected layers
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def backward(self, target, value):
        loss = self.loss(target, value).to(self.device)
        loss.backward()
        self.optimizer.step()
