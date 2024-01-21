# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import torch.nn as nn

from torchvision import models

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Classifier(nn.Module):
    """
    A custom classifier based on the ResNet18 architecture for image classification.

    Parameters:
    - num_classes (int): The number of classes in the target dataset.

    Attributes:
    - resnet (torchvision.models.resnet.ResNet): The ResNet18 model with a modified classifier head.

    Methods:
    - __init__(self, num_classes): Initializes the Classifier with a pre-trained ResNet18 model.
    - forward(self, x): Performs a forward pass through the network.

    Examples:
    ```python
    # Create a Classifier with 10 output classes
    classifier = Classifier(num_classes=10)
    
    # Forward pass through the network
    inputs = torch.randn(1, 3, 224, 224)  # Example input with shape (batch_size, channels, height, width)
    outputs = classifier(inputs)
    ```
    """

    def __init__(self, num_classes):
        """
        Initializes the Classifier with a pre-trained ResNet18 model.

        Parameters:
        - num_classes (int): The number of classes in the target dataset.
        """
        super(Classifier, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the classifier head to match the number of classes in your dataset
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters:
        - x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: The output tensor representing class probabilities.
        """
        return self.resnet(x)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------