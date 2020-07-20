from __future__ import print_function
import numpy as np
import pickle
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu


BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225 # IMG_SIZE*IMG_SIZE
P = 75 # D /// 3
LEARNING_RATE = 0.001


class EncDecNetLite():
    def __init__(self):
        super(EncDecNetLite, self).__init__()
        self.w_in = np.zeros((P, D))
        self.b_in = np.zeros((1, P))
        #
        # Please, add other weights here
        #

    def init(self):
        self.w_in = init_uniform(self.w_in)
        #
        # Please, add initializations of other weights here
        #

    def forward(self, x):
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P)) # [20, 75]
        a_in = np.matmul(x, self.w_in.transpose()) + B_in # [20, 75]
        z_in_numpy = relu(a_in)
        #
        # Please, add forward pass here
        #
        return 0 # y


    def backprop(self, some_args):
        #
        # Please, add backpropagation pass here
        #
        return 0 # dw

    def apply_dw(self, dw):
        #
        # Correct neural network''s weights
        #
        pass


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

# Main cycle
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(some_args)

    # Correct neural network''s weights
    neural_network.apply_dw(dw)

#
# Load images_test.pickle here, run the network on it and show results here
#
