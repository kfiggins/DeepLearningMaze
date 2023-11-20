import torch.nn as nn

# Define the neural network class 'BlobNet', which inherits from nn.Module
class BlobNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlobNet, self).__init__()

        # Define the layers of the network
        self.layer = nn.Sequential(
            # First layer: Linear transformation from input_size to hidden_size
            nn.Linear(input_size, hidden_size),
            # Activation function: ReLU (Rectified Linear Unit)
            nn.ReLU(),
            # Second layer: Linear transformation from hidden_size to output_size
            nn.Linear(hidden_size, output_size)
        )

    # Define the forward pass of the network
    def forward(self, x):
        # Pass the input 'x' through the sequential layers defined in __init__
        return self.layer(x)
