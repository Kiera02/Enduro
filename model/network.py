#!/bin/python3

# External
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learnable parameter for scaling the self-attention value
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax layer to compute attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            x: Input feature maps (Batch size x Channels x Width x Height)
        Returns:
            out: Self-attention value added to input feature
            attention: Attention scores
        '''

        batch_size, channels, width, height = x.size()

        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x

        return out, attention

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_width, input_height):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = SelfAttention(256)

        self.fc = nn.Linear(256 * (input_width // 8) * (input_height // 8), num_classes)

    def forward(self, x: torch.Tensor):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x, _ = self.attention(x)
        print(x.shape)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    cnn = CNN(in_channels=3, num_classes=10, input_width=64, input_height=64)
    print(cnn)
    input_data = torch.randn(1, 3, 64, 64)  # Example input data
    output = cnn.forward(input_data)  # Run the forward pass
    print(output)  # Print the output
