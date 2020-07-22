import numpy as np
import matplotlib.pyplot as plt


def diff_numpy(a, b, msg=None):
    """Shows differences between two tensors"""
    if a.shape != b.shape:
        print('Wrong shape!')
        print(a.shape)
        print(b.shape)
    else:
        diff = (np.sum(a - b))**2
        if msg:
            print('%s diff = %1.6f' % (msg, diff.item()))
        else:
            print('diff = %1.6f' % diff.item())


def images2batches(images):
    """Converts images to convenient for batching form"""
    ndata, img_size, _ = images.shape
    return np.reshape(images, (ndata, img_size*img_size))


def normalize(images):
    """Normalizes image pizel intensities to range from 0.0 to 1.0"""
    return images / 255.0


def imshow_side_by_side(images, titles):
    """Show image using matplotlib"""
    nrows = 1
    ncols = len(images)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4))
    for index, (image, title) in enumerate(zip(images, titles)):
        axes[index].imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
        axes[index].set_title(title)

    plt.show()


def plot_learning_curve(loss_values):
    """Plots learning curve using matplotlib"""
    plt.plot(list(range(len(loss_values))), loss_values)
    plt.xlabel('Updates no.')
    plt.ylabel('Sum squared loss')
    plt.title('Learning curve')
    plt.show()


def init_uniform(n_out, n_in):
    """Makes uniform initialization of weight matrix (please, use
    numpy.random.uniform function or similar"""

    return np.random.randn(n_out, n_in) * np.sqrt(2 / (n_in + n_out))


def relu(a):
    """Implements ReLU activation function"""
    return np.maximum(0, a)


def drelu(a):
    """Implements ReLU's derivative"""
    return (a > 0).astype(np.float)


def identity(a):
    """Implements Identity activation function"""
    return a


def didentity(a):
    """Implements Identity's derivative"""
    return np.ones_like(a)


def get_random_batch(batches_train, batch_size):
    """Outputs random batch of batch_size"""

    ndata, ndin = batches_train.shape
    indices = np.random.choice(ndata, batch_size, replace=False)

    return batches_train[indices]


def get_loss(Y_batch, X_batch_train):
    """Calculates sum squared loss"""
    return np.sum((Y_batch - X_batch_train)**2) / 2
