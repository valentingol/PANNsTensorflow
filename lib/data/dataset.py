import tensorflow as tf
import numpy as np
import os
import h5py

class AudioDataset:
    """Dataset class for GTZAN dataset.
    """
    def __init__(self, holdout_fold: int=2,shuffle:bool=True, 
                 sample_rate: int=32000,max_time: int=30, 
                 path: str='datasets/genres', h5py_path: str=None):
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
            Path of data folder of GTZAN,
            by default 'datasets/genres'.
        h5py_path : str or None, optional
            Path of h5 file if it exist. Otherwise, data will
            be loaded !![where?], by default None.
        """

        self.holdout_fold = holdout_fold
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.clip_samples = max_time * sample_rate
        self.path = path
        self.lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
        self.idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
        self.classes_num = len(labels)
        self.packed_hdf5_path = h5py_path

        print(f"Loading format is {loading_format}.")

        if loading_format == ".h5py" and self.packed_hdf5_path is None:
            self.packed_hdf5_path = os.path.join(self.path, "GTZAN_dataset.h5")
            print(f"Convert audio files to {loading_format} in "
                  f"{self.packed_hdf5_path}...")
            self.pack_audio_files_to_hdf5()
        else:
            "!![handle case loading_format != '.h5py'"

    def train_generator(self):
        """ !![one-line summary] """
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            while True: # !![comment]
                for i in self.indexes_train:
                    audio_name = f['audio_name'][i]
                    waveform = tf.expand_dims(
                                    self.int16_to_float32(f["waveform"][i]),
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
        """ !![one-line summary] """
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            for i in self.indexes_val:
                audio_name = f["audio_name"][i]
                waveform = tf.expand_dims(
                               self.int16_to_float32(f["waveform"][i]),
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

        with h5py.File(self.packed_hdf5_path, 'r') as hf:
            folds = tf.cast(hf['fold'][:], tf.int64)
        self.holdout_fold = tf.cast(self.holdout_fold, tf.int64)
        self.indexes_train = tf.where(folds != self.holdout_fold)[:, 0]
        self.indexes_val = tf.where(folds == self.holdout_fold)[:, 0]
        # !![w is not explicit: change name]
        self.w = tf.where(folds == self.holdout_fold)
        if self.shuffle:
            tf.random.set_seed(self.random_seed)
            self.indexes_train = tf.random.shuffle(self.indexes_train)
            self.indexes_val = tf.random.shuffle(self.indexes_val)

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
        # prepare data
        train_ds = train_ds.batch(self.batch_size).cache()
        val_ds = val_ds.batch(self.batch_size).cache()
        return train_ds.batch(self.batch_size), val_ds.batch(self.batch_size)


dataset = DatasetGTZAN(augmentation=augmentation,
                       h5py_path="datasets\genres\GTZAN_dataset.h5")
train_ds, val_ds = dataset()