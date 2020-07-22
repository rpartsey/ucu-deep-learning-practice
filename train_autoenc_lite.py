from __future__ import print_function
import time
import pickle
import numpy as np
from tqdm import tqdm
from utils import (
    init_uniform, get_random_batch, images2batches, normalize, relu, identity,
    drelu, didentity, get_loss, imshow_side_by_side, plot_learning_curve
)


BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225  # IMG_SIZE * IMG_SIZE
P = 75  # D // 3
LEARNING_RATE = 0.001

np.random.seed(0)  # Set the random seed to make experiments reproducible


class EncDecNetLite:
    def __init__(self):
        # Layer_in
        self.w_in = np.zeros([P, D])
        self.b_in = np.zeros([1, P])
        # Layer_rec
        self.w_rec = np.zeros([P, P])
        self.b_rec = np.zeros([1, P])
        # Layer_link
        self.w_reduce = np.zeros([D, P])
        self.w_link = np.zeros([P, P])
        # Layer_out
        self.w_out = np.zeros([D, P])
        self.b_out = np.zeros([1, D])
        # Store EncDecNetLite's state
        self._state = {}

    def init(self):
        """Weights initialization"""

        self.w_in = init_uniform(*self.w_in.shape)
        self.w_link = init_uniform(*self.w_link.shape)
        self.w_out = init_uniform(*self.w_out.shape)
        self.w_rec = np.eye(*self.w_rec.shape)
        self.w_reduce = np.hstack(
            [np.arange(self.w_reduce.shape[0]).reshape([-1, 1]) == 3 * i
             for i in range(self.w_reduce.shape[1])]
        ).astype(np.uint8)

    def forward(self, x):
        """Forwardpropagation pass"""

        # Layer_in
        a_in = np.matmul(x, self.w_in.T) + self.b_in
        z_in = relu(a_in)
        # Layer_rec
        a_rec = np.matmul(z_in, self.w_rec.T) + self.b_rec
        z_rec = relu(a_rec)
        # Layer_link
        a_link = np.matmul(np.matmul(x, self.w_reduce), self.w_link.T)
        z_link = identity(a_link)
        # Layer_out
        a_out = np.matmul(z_rec+z_link, self.w_out.T) + self.b_out
        y = relu(a_out)

        # Set internal state
        self._state.update({
            'x': x,
            'y': y,
            'a_out': a_out,
            'z_link': z_link,
            'a_link': a_link,
            'z_rec': z_rec,
            'a_rec': a_rec,
            'z_in': z_in,
            'a_in': a_in,
        })

        return y

    def backprop(self):
        """Backpropagation pass"""

        # Layer_out
        x = self._state['x']
        y = self._state['y']
        a_out = self._state['a_out']
        z_rec = self._state['z_rec']
        z_link = self._state['z_link']

        delta_out = (y-x) * drelu(a_out)
        dw_out = np.matmul(delta_out.T, z_rec + z_link)
        db_out = np.sum(delta_out, axis=0, keepdims=True)

        # Layer_link
        a_link = self._state['a_link']

        delta_link = didentity(a_link) * np.matmul(delta_out, self.w_out)
        dw_link = np.matmul(delta_link.T, np.matmul(x, self.w_reduce))

        # Layer_in
        a_in = self._state['a_in']

        delta_in = drelu(a_in) * np.matmul(delta_out, self.w_out)
        dw_in = np.matmul(delta_in.T, x)
        db_in = np.sum(delta_in, axis=0, keepdims=True)

        return {
            'w_in': dw_in,
            'b_in': db_in,
            'w_link': dw_link,
            'w_out': dw_out,
            'b_out': db_out,
        }

    def apply_dw(self, d):
        """Update neural network's weights"""
        self.w_in -= LEARNING_RATE * d['w_in']
        self.b_in -= LEARNING_RATE * d['b_in']
        self.w_link -= LEARNING_RATE * d['w_link']
        self.w_out -= LEARNING_RATE * d['w_out']
        self.b_out -= LEARNING_RATE * d['b_out']


# Load train data
with open('images_train.pickle', 'rb') as file_train:  # use 'with' context manager to automatically close opened file
    images_train = pickle.load(file_train)

# Convert images to batching-friendly format
batches_train = normalize(images2batches(images_train))

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

# Main cycle
loss_history = []
for i in tqdm(range(UPDATES_NUM), desc='Training loop'):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)
    loss_history.append(loss)

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop()

    # Correct neural network's weights
    neural_network.apply_dw(dw)

# Show learning curve
plot_learning_curve(loss_history)

# Load images_test.pickle here, run the network on it and show results
with open('images_test.pickle', 'rb') as file_test:
    images_test = pickle.load(file_test)

batch_test = normalize(images2batches(images_test))

for image in batch_test:
    imshow_side_by_side(
        images=[image.reshape([IMG_SIZE, IMG_SIZE]), neural_network.forward(image).reshape([IMG_SIZE, IMG_SIZE])],
        titles=['Ground truth', 'Autoencoded']
    )

# Scalar vs vectorized Layer_in forward path execution time comparison
x = np.random.randn(BATCH_SIZE, D)
w_in = init_uniform(P, D)
b_in = np.zeros([1, P])

# Scalar:
scalar_start = time.time()

a_in_scalar = np.zeros([BATCH_SIZE, P])

for k in range(BATCH_SIZE):
    for i in range(P):
        for j in range(D):
            a_in_scalar[k, i] += x[k, j] * w_in[i, j]
        a_in_scalar[k, i] += b_in[0, i]
z_in_scalar = relu(a_in_scalar)

print(f'Layer_in scalar forward path execution time: {time.time() - scalar_start} seconds.')

# Vectorized:
vector_start = time.time()

a_in_vectorized = np.matmul(x, w_in.T) + b_in
z_in_vectorized = relu(a_in_vectorized)

print(f'Layer_in vectorized forward path execution time: {time.time() - vector_start} seconds elapsed.')

print('Z_in_scalar:\n', z_in_scalar)
print('Z_in_vectorized:\n', z_in_vectorized)

assert np.allclose(z_in_scalar, z_in_vectorized), "Postsynaptic values aren't the same"