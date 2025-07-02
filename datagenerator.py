import numpy as np
import tensorflow as tf
from keras.utils import Sequence
import random

class datagen(Sequence):
    def __init__(self, file_paths, labels, batch_size=4, shuffle=True, augment=False):
        """
        Args:
            file_paths (list of str): Paths to input volumes
            labels (list of int): Corresponding binary labels (0 or 1)
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle data between epochs
            augment (bool): Whether to apply random augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
    
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        # Generate indexes for the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Select data and labels
        batch_file_paths = [self.file_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]
        
        # Generate data
        X, y = self.__data_generation(batch_file_paths, batch_labels)
        return X, y

    def __data_generation(self, batch_file_paths, batch_labels):
        # Initialize empty arrays
        X = np.empty((len(batch_file_paths), 240, 240, 160, 3), dtype=np.float16)
        y = np.empty((len(batch_file_paths), 1), dtype=np.float16)
        
        for i, path in enumerate(batch_file_paths):
            # Load the volume (customize this depending on your data format)
            volume = np.load(path)  # assuming .npy format
            if self.augment:
                volume = self.__augment(volume)
            X[i,] = volume
            y[i] = batch_labels[i]
        
        return X, y

    def __augment(self, volume):
        # Basic augmentation (flip, noise, etc.)
        if random.random() > 0.5:
            volume = np.flip(volume, axis=0)
        if random.random() > 0.5:
            volume = np.flip(volume, axis=1)
        if random.random() > 0.5:
            volume += np.random.normal(0, 0.01, volume.shape)
        return volume

