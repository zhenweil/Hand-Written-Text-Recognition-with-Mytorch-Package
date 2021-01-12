import sys
import numpy as np

from mytorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    Args:
        params (list or iterator): <some module>.parameters()
        lr (float): learning rate (eta)
        momentum (float): momentum factor (beta)

    Inherits from:
        Optimizer (optim.optimizer.Optimizer)
    """
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params) # inits parent with network params
        self.lr = lr
        self.momentum = momentum

        # This tracks the momentum of each weight in each param tensor
        self.momentums = [np.zeros(t.shape) for t in self.params]

    def step(self):
        """Updates params based on gradients stored in param tensors"""
        length = len(self.params)
        for i in range(length):
            param = self.params[i]
            prev_W = param.data
            grad = param.grad.data
            momentum_grad = self.momentums[i]
            d_W = self.momentum * momentum_grad - self.lr*grad
            curr_W = prev_W + d_W
            self.params[i].data = curr_W
            self.momentums[i] = d_W
        self.zero_grad()
