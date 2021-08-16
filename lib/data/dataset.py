import tensorflow as tf
import numpy as np
import os
import h5py
from utils.prepare_data import int16_to_float32

class AudioDataset:
    """Dataset class for Ausio datasets."""

    def __init__(self, holdout_fold: int=2,shuffle:bool=True, 
                 sample_rate: int=32000, max_time: int=30, 
                 path: str='audioset.h5'):
        """
        Parameters
        ----------
        holdout_fold : int, optional
            Validation data fold, by default 2. !![more comments]
        shuffle : bool, optional
            Whether sata will be shuffle or not, by default True.
        sample_rate : int, optional)
            Sample rate of inputs, by default 32000.
        max_time : int, optional
            Time of waveform sequence. Sequences are padded or
            truncated to max_time lenght, by default 30.
        path : str, optional
            Path of '.h5' file where custom data are strored,
            Use preprocessing to generate such a file
            by default 'audioset.h5'.
        """
        self.holdout_fold = holdout_fold
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.clip_samples = max_time * sample_rate
        self.path = path
        with h5py.File(self.path, 'r') as hf:
            self.classes_num = tf.cast(hf['classes_num'], tf.int64)
        
        # self.lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
        # self.idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
        # self.classes_num = len(labels)

    def train_generator(self):
        """ Generator feeding training dataset """
        with h5py.File(self.path, 'r') as f:
            # allow to run through dataset circularly
            while True:
                for i in self.indexes_train:
                    audio_name = f['audio_name'][i]
                    waveform = tf.expand_dims(
                                    int16_to_float32(f["waveform"][i]),
                                    axis=-1
                                    )
                    target = f["target"][i]
                    yield {"audio_name": audio_name,
                           "waveform": waveform,
                           "target": target}
                if self.shuffle:
                    tf.random.set_seed(self.random_seed)
                    self.indexes_train = tf.random.shuffle(self.indexes_train)

    def val_generator(self):
        """ Generator feeding validation dataset """
        with h5py.File(self.path, 'r') as f:
            # run through dataset only one time
            for i in self.indexes_val:
                audio_name = f["audio_name"][i]
                waveform = tf.expand_dims(
                               int16_to_float32(f["waveform"][i]),
                               axis=-1
                               )
                target = f["target"][i]
                yield {"audio_name": audio_name,
                      "waveform": waveform,
                      "target": target}

    def __call__(self, random_seed: int=None):
        """Create batched train and validation datasets

        Parameters
        ----------
        random_seed : int or None, optional
            Random seed, by default None.

        Returns
        -------
        train_dataset : dict tf.data.Dataset
            Dataset containing dictionnary of format:
            {"audio_names": tf.Tensor, shape=(batch_size,),
             "waveform": tf.Tensor, shape=(batch_size, clip_samples, 1),
             "target": tf.Tensor, shape=(batch_size, classes_num)}

        val_dataset : dict tf.data.Dataset
            Dataset containing dictionnary of format:
            {"audio_names": tf.Tensor, shape=(batch_size,),
             "waveform": tf.Tensor, shape=(batch_size, clip_samples, 1),
             "target": tf.Tensor, shape=(batch_size, classes_num)}
        """
        self.random_seed = random_seed

        #Creation of a balanced validation set
        with h5py.File(self.path, 'r') as hf:
            folds = tf.cast(hf['fold'][:], tf.int64)
        self.holdout_fold = tf.cast(self.holdout_fold, tf.int64)
        self.indexes_train = tf.where(folds != self.holdout_fold)[:, 0]
        self.indexes_val = tf.where(folds == self.holdout_fold)[:, 0]

        # shuffle data if shuffle==True
        if self.shuffle:
            tf.random.set_seed(self.random_seed)
            self.indexes_train = tf.random.shuffle(self.indexes_train)
            self.indexes_val = tf.random.shuffle(self.indexes_val)

        # Train Dataset creation
        train_ds = tf.data.Dataset.from_generator(
                    self.train_generator,
                    output_signature=({
                    "audio_name": tf.TensorSpec(shape=(), dtype=tf.string),
                    "waveform": tf.TensorSpec(shape=(self.clip_samples, 1),
                                              dtype=tf.float32),
                    "target": tf.TensorSpec(shape=(self.classes_num),
                                            dtype=tf.float32)
                    })
                    )

        # Validation Dataset creation
        val_ds = tf.data.Dataset.from_generator(
                self.val_generator,
                output_signature=({
                "audio_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "waveform": tf.TensorSpec(shape=(self.clip_samples, 1),
                                          dtype=tf.float32),
                "target": tf.TensorSpec(shape=(self.classes_num),
                                        dtype=tf.float32)
                })
                )

        return train_ds, val_ds


dataset = DatasetGTZAN(augmentation=augmentation,
                       h5py_path="datasets\genres\GTZAN_dataset.h5")
train_ds, val_ds = dataset()