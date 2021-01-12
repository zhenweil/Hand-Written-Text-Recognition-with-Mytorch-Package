import numpy as np
import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T), None

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data), None

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
            
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad = requires_grad,
                          is_leaf = not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -np.ones(b.shape) * grad_output.data
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data,b.data), requires_grad = requires_grad, 
                          is_leaf = not requires_grad)
        return c
    def backward(ctx, grad_output):
        a,b = ctx.saved_tensors
        grad_a = np.multiply(b.data, grad_output.data)
        grad_b = np.multiply(grad_output.data, a.data)
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data, 1/b.data), requires_grad = requires_grad,
                          is_leaf = not requires_grad)
        return c
    def backward(ctx, grad_output):
        a,b = ctx.saved_tensors
        grad_a = np.multiply(1/b.data, grad_output.data)
        grad_b_temp = np.multiply(grad_output.data, a.data)
        grad_b = np.multiply(grad_b_temp, -1/((b.data)**2))
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)
            
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data @ b.data, requires_grad = requires_grad,
                          is_leaf = not requires_grad)
        return c
    def backward(ctx, grad_output):
        a,b = ctx.saved_tensors
        grad_a = np.dot(grad_output.data, b.data.T)
        grad_b = np.dot(a.data.T, grad_output.data)
        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)
class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad = requires_grad, is_leaf = not requires_grad)
        return c
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_a = np.multiply(grad_output.data, np.exp(a.data))
        grad_a = unbroadcast(grad_a, a.shape)
        return tensor.Tensor(grad_a), None

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only tensor can be passed to ReLU")
        data = a.data
        data[data <= 0] = 0
        ctx.save_for_backward(a)
        ctx.output = data
        return tensor.Tensor(data, requires_grad = a.requires_grad,
                          is_leaf = a.is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        curr_gradient = ctx.output
        curr_gradient[curr_gradient > 0] = 1
        curr_gradient[curr_gradient <= 0] = 0
        gradient_a = np.multiply(grad_output.data, curr_gradient)
        return tensor.Tensor(gradient_a), None

def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    
    batch_size, num_classes = predicted.shape
    label = to_one_hot(target, num_classes) # Convert target to one hot data (numpy)
    const = np.max(predicted.data)
    const = tensor.Tensor(const, is_parameter = False) # Define the constant to do exp trick
    predicted_normalized = predicted.__sub__(const)
    predicted_normalized_exp = predicted_normalized.__exp__()
    predicted_normalized_exp_sum = predicted_normalized_exp.__sum__(axis = 1, keepdims = False)
    log_predicted_normalized_exp_sum = predicted_normalized_exp_sum.__log__()
    log_predicted_exp_sum = log_predicted_normalized_exp_sum.__add__(const)
    log_predicted_exp_sum_reshaped = log_predicted_exp_sum.reshape(log_predicted_exp_sum.shape[0],1)
    LogSoftmax = predicted.__sub__(log_predicted_exp_sum_reshaped)
    XE = LogSoftmax.__mul__(label)
    XE_sum = XE.__sum__(axis = 0, keepdims = False)
    XE_negative = XE.__mul__(tensor.Tensor(-1, is_parameter = False))
    XE_negative_sum = XE_negative.__sum__(axis = 0, keepdims = False)
    XE_negative_sum = XE_negative_sum.__sum__(axis = 0, keepdims = False)
    NLLLoss = XE_negative_sum.__div__(tensor.Tensor(batch_size, is_parameter = False))
    return NLLLoss

def to_one_hot(arr, num_classes):
    """
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)
