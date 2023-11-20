import torch.nn as nn

class BlobNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BlobNet, self).__init__()

        # Define the layers of the network
        self.layer = nn.Sequential(
            # First layer: Linear transformation from input_size to hidden_size1
            nn.Linear(input_size, hidden_size1),
            # Activation function: ReLU (Rectified Linear Unit)
            nn.ReLU(),
            # Second layer: Linear transformation from hidden_size1 to hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            # Another ReLU activation function
            nn.ReLU(),
            # Final layer: Linear transformation from hidden_size2 to output_size
            nn.Linear(hidden_size2, output_size)
        )

    # Define the forward pass of the network
    def forward(self, x):
        # Pass the input 'x' through the sequential layers defined in __init__
        return self.layer(x)
