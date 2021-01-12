import sys
sys.path.append("..")
import numpy as np
from mytorch.nn.activations import *
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor
from mytorch.nn.loss import CrossEntropyLoss
from find_letters import get_letter_from_img
from scipy import io
import matplotlib.pyplot as plt
import string

BATCH_SIZE = 50

def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs = 3):
    val_accuracies = []
    num_data = train_x.shape[0]
    num_feature = train_x.shape[1]
    train_y = train_y.reshape(train_y.shape[0],1)
    train_data = np.hstack((train_x, train_y))
    num_batches = num_data//BATCH_SIZE
    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        for i in range(num_batches):
            batch = train_data[BATCH_SIZE*i:BATCH_SIZE*(i+1),:]
            batch_x = batch[:,:num_feature]
            batch_y = batch[:,-1]
            batch_x_tensor = Tensor(batch_x, requires_grad = False)
            batch_y_tensor = Tensor(batch_y, requires_grad = False)
            optimizer.zero_grad()
            out = model(batch_x_tensor)
            loss = criterion(out, batch_y_tensor)
            loss.backward()
            optimizer.step()
            
            if(i % 10 == 0):
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                print("epoch: {}, loss: {}, accuracy: {}".format(epoch, loss.data, accuracy))

    return val_accuracies

def validate(model, val_x, val_y):
    num_val = val_x.shape[0]
    num_feature = val_x.shape[1]
    val_y = val_y.reshape(val_y.shape[0],1)
    val_data = np.hstack((val_x, val_y))
    num_batches = num_val//BATCH_SIZE
    for i in range(num_batches):
        batch = val_data[BATCH_SIZE*i:BATCH_SIZE*(i+1),:]
        batch_x = batch[:,:num_feature]
        batch_y = batch[:,-1]
        batch_x_tensor = Tensor(batch_x)
        batch_y_tensor = Tensor(batch_y)
        out = model(batch_x_tensor)
        out_np = out.data
        batch_preds = np.argmax(out_np, axis = 1)
        num_correct = np.zeros(batch_y.shape)
        num_correct[batch_preds == batch_y] = 1
        num_correct = int(np.sum(num_correct))
        accuracy = num_correct/batch_y.shape[0]
    return accuracy

def test(model, spaces, test_data):
    results = []
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    num_rows = len(test_data)
    for row in range(num_rows):
        row_result = []
        row_space = spaces[row]
        row_data = test_data[row]
        input_data = np.asarray(row_data)
        input_data = Tensor(input_data)
        out = model(input_data)
        out = out.data
        prediction = np.argmax(out, axis = 1)
        prediction = letters[prediction]
        prediction = list(map(str, prediction))
        prediction_space_seperated = ''.join(map(''.join, zip(row_space, prediction))) # Convert prediction to string along with blanks
        print(prediction_space_seperated)
        results.append(prediction_space_seperated)
    return results
def main():
    lr = 0.01
    # Load data
    train_data = io.loadmat('../data/nist36_train.mat')
    valid_data = io.loadmat('../data/nist36_valid.mat')
    train_x = train_data['train_data']
    train_y = train_data['train_labels']
    train_y = np.argmax(train_y, axis = 1)
    valid_x = valid_data['valid_data']
    valid_y = valid_data['valid_labels']
    valid_y = np.argmax(valid_y, axis = 1)

    # Define model
    model = Sequential(Linear(1024, 128), ReLU(),
                       Linear(128, 36))
    optimizer = SGD(model.parameters(), lr = lr, momentum = 0.9)
    criterion = CrossEntropyLoss()

    train(model, optimizer, criterion, train_x, train_y, valid_x, valid_y, num_epochs = 3)
    spaces, test_data = get_letter_from_img("../images/test.jpg")
    test(model, spaces, test_data)
if __name__ == '__main__':
    main()