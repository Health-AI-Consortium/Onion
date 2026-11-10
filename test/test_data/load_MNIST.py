# courtesy of https://www.kaggle.com/code/hojjatk/read-mnist-dataset
import os
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import argparse
from test.test_data.download_MNIST import download_MNIST

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)



#
# MNIST Partitioning (for federated learning test setups)
#
_MNIST_DIR = join(os.path.dirname(os.path.realpath(__file__)), "MNIST")


def _load_mnist():
    paths = {
        "train": (join(_MNIST_DIR, "train-images.idx3-ubyte"), join(_MNIST_DIR, "train-labels.idx1-ubyte")),
        "test": (join(_MNIST_DIR, "t10k-images.idx3-ubyte"), join(_MNIST_DIR, "t10k-labels.idx1-ubyte")),
    }
    missing = [path for pair in paths.values() for path in pair if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing MNIST files: {}. Run test/test_data/download_MNIST.py first.".format(", ".join(missing))
        )
    (x_train, y_train), (x_test, y_test) = MnistDataloader(*paths["train"], *paths["test"]).load_data()
    return (
        np.array(x_train, dtype=np.uint8),
        np.asarray(y_train, dtype=np.uint8),
        np.array(x_test, dtype=np.uint8),
        np.asarray(y_test, dtype=np.uint8),
    )


def _partition_indices(n: int, num_partitions: int, seed: int):
    perm = np.random.default_rng(seed).permutation(n)
    indices = []
    start = 0
    for i in range(num_partitions):
        size = n // num_partitions + (1 if i < n % num_partitions else 0)
        indices.append(perm[start:start + size])
        start += size
    return indices


def _write_images(filepath, images):
    with open(filepath, "wb") as file:
        file.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        file.write(images.astype(np.uint8).tobytes())


def _write_labels(filepath, labels):
    with open(filepath, "wb") as file:
        file.write(struct.pack(">II", 2049, len(labels)))
        file.write(labels.astype(np.uint8).tobytes())


def partition_MNIST(num_partitions: int, seed: int = 7, save_dir: str = None):
    """Split the MNIST train and test sets into num_partitions disjoint shards and save each
    shard as its own dataset (IDX files) that can be loaded with MnistDataloader.

    Returns the list of partition directories, one per client.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")
    if save_dir is None:
        save_dir = join(_MNIST_DIR, "partitions")

    x_train, y_train, x_test, y_test = _load_mnist()
    train_idx = _partition_indices(len(x_train), num_partitions, seed)
    test_idx = _partition_indices(len(x_test), num_partitions, seed)

    partitions = []
    for i in range(num_partitions):
        partition_dir = join(save_dir, "partition_{}".format(i))
        os.makedirs(partition_dir, exist_ok=True)
        _write_images(join(partition_dir, "train-images.idx3-ubyte"), x_train[train_idx[i]])
        _write_labels(join(partition_dir, "train-labels.idx1-ubyte"), y_train[train_idx[i]])
        _write_images(join(partition_dir, "test-images.idx3-ubyte"), x_test[test_idx[i]])
        _write_labels(join(partition_dir, "test-labels.idx1-ubyte"), y_test[test_idx[i]])
        partitions.append(partition_dir)
    return partitions


def get_MNIST_partition(partition_id: int, num_partitions: int, seed: int = 7):
    """Return the in-memory shard for client partition_id, in the same format as
    MnistDataloader.load_data(): ((x_train, y_train), (x_test, y_test)) as numpy arrays.

    Uses the same partitioning as partition_MNIST, so a client gets exactly the same
    samples here as from the saved partition files.
    """
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be in [0, {})".format(num_partitions))
    x_train, y_train, x_test, y_test = _load_mnist()
    train_idx = _partition_indices(len(x_train), num_partitions, seed)[partition_id]
    test_idx = _partition_indices(len(x_test), num_partitions, seed)[partition_id]
    return (x_train[train_idx], y_train[train_idx]), (x_test[test_idx], y_test[test_idx])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default= 7)
    # parser.add_argument("--id", default = 0)
    parser.add_argument("--partitions", default = 4)

    args = parser.parse_args()

    mnist_path = download_MNIST()
    partitions = partition_MNIST(num_partitions = args.partitions, seed = args.seed)


    # x_train, y_train, x_test, y_test = get_MNIST_partition(args.id, args.partitions, args.seed)
   