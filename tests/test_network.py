# Internal
import os
import sys

# External
import cv2
import torch

# Path Append
sys.path.append(os.path.abspath(os.curdir))

# Internal
from model.network import CNN

def test_cnn_with_real_image():
    # Load an image using OpenCV
    image_path = 'tests/enduro.png'
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    assert image is not None, "Image not found or path is incorrect"

    # Convert image from BGR to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the expected input size of the network
    input_width, input_height = image.shape[:2]

    # Normalize and convert to torch tensor
    # Change data layout from HxWxC to CxHxW
    image = image.transpose(2, 0, 1)
    # Scale to [0, 1]
    image = torch.from_numpy(image).float() / 255.0
    # Add batch dimension
    image = image.unsqueeze(0)

    # Parameters for the CNN
    # Number of input channels (RGB)
    in_channels = 3
    # Number of output classes
    num_classes = 10

    # Instantiate the CNN
    model = CNN(in_channels, num_classes, input_width, input_height)

    # Forward pass through the model
    output = model(image)

    # Check the output shape
    expected_shape = (1, num_classes)
    assert output.shape == expected_shape, "Output shape is incorrect"
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"
