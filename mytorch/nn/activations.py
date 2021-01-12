import mytorch.nn.functional as F
from mytorch.nn.module import Module


class ReLU(Module):
    """ReLU activation function
    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ReLU forward pass.

        Args:
            x (Tensor): input before ReLU
        Returns:
            Tensor: input after ReLU
        """
        return F.ReLU.apply(x)