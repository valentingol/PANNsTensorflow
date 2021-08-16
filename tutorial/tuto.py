import os
import time
from typing import List

import h5py
import librosa
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tqdm import tqdm

#%% Mixup
augmentation = ['mixups']

class Mixup(object):
    def __init__(self, mixup_alpha: float, random_seed:int =None):
        """
        Parameters
        ----------
        mixup_alpha: float
            !![description]
        random_seed: int or None, optional
            !![description], by default None.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size: int):
        """Get mixup random coefficients.

        Parameters
        ----------
        batch_size: int
            Size of batch.

        Returns
        -------
        mixup_lambdas: tf.Tensor, shape=(batch_size,)
            !![description]
        """
        mixup_lambdas = []
        for _ in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha,
                                         self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)

        return np.array(mixup_lambdas)


def do_mixup(x: tf.Tensor, mixup_lambda: tf.Tensor):
    """Mixup on 'x' of even indexes (0, 2, 4, ...) with
    odd indexes (1, 3, 5, ...).

    Parameters
    ----------
    x: tf.Tensor, shape=(batch_size * 2, ...)
        !![description]
    mixup_lambda: tf.Tensor, shape=(batch_size * 2,)
        !![description]

    Returns
    -------
    out: tf.Tensor, shape=(batch_size, ...)
        !![description]
    """
    # !![comment]
    transpose = lambda x: tf.einsum("i...j->j...i", x)
    even_ind = transpose(x[0 :: 2]) * mixup_lambda[0 :: 2]
    odd_ind = transpose(x[1 :: 2]) * mixup_lambda[1 :: 2]
    out = transpose(even_ind + odd_ind)
    return out


#%% Dataset
class DatasetGTZAN:
    """Dataset class for GTZAN dataset.
    """
    def __init__(self, batch_size: int=8, holdout_fold: int=2,
                 shuffle:bool=True, sample_rate: int=32000, max_time: int=30,
                 path: str='datasets/genres', loading_format: str=".h5py",
                 h5py_path: str=None, augmentation: List[str]=["mixup"],
                 labels: List[str]=['blues', 'classical', 'country', 'disco',
                            'hiphop','jazz', 'metal', 'pop', 'reggae', 'rock']
                 ):
        """
        Parameters
        ----------
        batch_size : int, optional
            Size of batch, by default 8. !![8 small?]
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
        loading_format str, optional
            One of '.h5py', !![complete list?]
            Data loading format, by default ".h5py".
        h5py_path : str or None, optional
            Path of h5 file if it exist. Otherwise, data will
            be loaded !![where?], by default None.
        augmentation : str list, optional
            List of augmentation. If "mixup" in augmentation,
            train_batches size will be multiply by 2 to
            process mixup (effective batchsize unchanged),
            by default ["mixup"].
        labels : str list, optional
            Labels of dataset, by default ['blues', 'classical',
            'country', 'disco','hiphop', 'jazz', 'metal', 'pop',
            'reggae', 'rock'].
        """
        if 'mixup' in augmentation:
            self.batch_size = 2 * batch_size
        else:
            self.batch_size = batch_size

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

    @staticmethod
    def browse_folder(fd: str): # !![fd not enough explicit: change name]
        """ !![one-line summary] """
        paths, names = [], []
        for root, _, files in os.walk(fd):
            for name in files:
                if name.endswith('.wav'):
                    filepath = os.path.join(root, name)
                    names.append(name)
                    paths.append(filepath)
        return names, paths

    @staticmethod
    def float32_to_int16(x: tf.Tensor):
        """ !![one-line summary] """
        maxi = tf.reduce_max(tf.abs(x))
        if maxi > 1.0: # !![why?: comment]
            x /= maxi
        return tf.cast(x * (2.0**15 - 1), tf.int16)

    def to_one_hot(self, k: int):
        """ !![one-line summary] """
        target = tf.zeros(self.classes_num, dtype="int!![complete]")
        target[k] = 1
        return target

    def pad_truncate_sequence(self, x: tf.Tensor):
        """ !![one-line summary, precise that x is an 1-D tensor] """
        if len(x) < self.clip_samples:
            return tf.concatenate([x, np.zeros(self.clip_samples - len(x))])
        else:
            return x[0:self.clip_samples]

    def get_target(self, audio_name: str):
        """ !![one-line summary] """
        return self.lb_to_idx[audio_name.split('.')[0]]

    def pack_audio_files_to_hdf5(self):
        """ !![one-line summary] """
        # Define paths
        audios_dir = os.path.join(self.path)
        if not self.packed_hdf5_path.endswith('.h5'):
            self.packed_hdf5_path += '.h5'
        if os.path.exists(self.packed_hdf5_path):
            os.remove(self.packed_hdf5_path)
        if os.path.dirname(self.packed_hdf5_path) != '':
            os.makedirs(os.path.dirname(self.packed_hdf5_path), exist_ok=True)

        (audio_names, audio_paths) = self.traverse_folder(audios_dir)

        audio_names = sorted(audio_names)
        audio_paths = sorted(audio_paths)
        audios_num = len(audio_names)

        # targets are found using get_target
        targets = [self.get_target(audio_name) for audio_name in audio_names]

        meta_dict = {
            'audio_name': np.array(audio_names),
            'audio_path': np.array(audio_paths),
            'target': tf.convert_to_tensor(np.array(targets)), # !![convert_to_tensor??]
            'fold': tf.convert_to_tensor(np.arange(len(audio_names)) % 10 + 1)} # !![convert_to_tensor??]

        feature_time = time.time()
        with h5py.File(self.packed_hdf5_path, 'w') as hf:
            hf.create_dataset(
                name='audio_name',
                shape=(audios_num,),
                dtype='S80')

            hf.create_dataset(
                name='waveform',
                shape=(audios_num, self.clip_samples),
                dtype=np.int16)

            hf.create_dataset(
                name='target',
                shape=(audios_num, self.classes_num),
                dtype=np.float32)

            hf.create_dataset(
                name='fold',
                shape=(audios_num,),
                dtype=np.int32)

            for i in tqdm(range(audios_num), total=audios_num):
                audio_name = meta_dict['audio_name'][i]
                fold = meta_dict['fold'][i]
                audio_path = meta_dict['audio_path'][i]
                target = meta_dict['target'][i]

                audio, _ = librosa.core.load(audio_path, sr=self.sample_rate,
                                               mono=True)
                audio = self.pad_truncate_sequence(audio)

                hf['audio_name'][i] = audio_name.encode() # !![why encode?]
                hf['waveform'][i] = self.float32_to_int16(audio)
                hf['target'][i] = self.to_one_hot(target)
                hf['fold'][i] = fold

        print(f'Write hdf5 to {self.packed_hdf5_path}')
        print(f'Time: {time.time() - feature_time:.3f} sec')

    @staticmethod
    def int16_to_float32(x):
        """ !![one-line summary] """
        return tf.cast(x, tf.float32) / (2.0**15 - 1.0)

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

#%% get datasets
dataset = DatasetGTZAN(augmentation=augmentation,
                       h5py_path="datasets\genres\GTZAN_dataset.h5")
train_ds, val_ds = dataset()
idx_to_lb = dataset.idx_to_lb

#%%
##############################################################################
########################### REQUIREMENTS #####################################
# model: tf.keras.Model
##############################################################################

# avoid linter triggering
model = None

## Build your own custom train loop
#%% Optimizer and compilation
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    amsgrad=True,
    name="Adam",
    )
model.compile(optimizer=optimizer)

#%% Loss (categorical cross-entropy)
def loss_func(output_dict, target_dict):
    f_mean = tf.reduce_mean
    loss = - f_mean(target_dict["target"] * output_dict["clipwise_output"])
    return loss

#%% Mix-up
if 'mixup' in augmentation:
    mixup_augmenter = Mixup(mixup_alpha=1.0)

#%% Checkpoints
checkpoints_dir = './checkpoints'
os.makedirs('checkpoints', exist_ok=True)

#%% Forward pass function for validation
def forward(model, dataset, return_input=False, return_target=False):
    """Forward data to a model.
    Args:
      model: tf.keras.Model
      dataset: tf.dataset.Dataset
      return_input: bool
      return_target: bool
    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    def append_to_dict(dictionary, key, value):
        if key in dictionary.keys():
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    output_dict = {}
    device = next(model.parameters()).device

    # Forward data to a model in mini-batches
    for batch_data_dict in dataset:
        batch_waveform = batch_data_dict['waveform']
        # forward pass in evaluation mode
        batch_output = model(batch_waveform, training=False)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])
        append_to_dict(output_dict, 'clipwise_output',
            batch_output['clipwise_output'].numpy())

        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

    # all ouputs are numpy arrays
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict

#%% Accuracy
def calculate_accuracy(y_true, y_score):
    """Calculate accuracy
    Args:
      y_true: np.array
      y_score: np.array
    Returns:
      accuracy: float
    """
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1))
    accuracy /= y_true.shape[0]
    return accuracy

#%% Training loop
init_iteration = 0
stop_iteration = 1000
validation_cycle = 50
checkpoint_cycle = 50
n_cycle = 5

# Train on mini batches
for _ in range(n_cycle):
    iteration = init_iteration
    for batch_data_dict in train_ds:

        print(f'Iteration {iteration}', end=', ')

        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                len(batch_data_dict['waveform'])
                )

        # Train
        with tf.GradientTape() as tape:
            if 'mixup' in augmentation:
                batch_output_dict = model(batch_data_dict['waveform'],
                    batch_data_dict['mixup_lambda'], training=True)
                batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                    batch_data_dict['mixup_lambda'])}
            else:
                batch_output_dict = model(batch_data_dict['waveform'], None)
                batch_target_dict = {'target': batch_data_dict['target']}

            # loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            print(f'loss: {loss}', end='')

        # Backward pass and update weights
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        if iteration == stop_iteration:
            break

        iteration += 1

    # Validation
    output_dict = forward(model, val_ds, return_target=True)
    # (all outputs are numpy arrays)
    clipwise_output = output_dict['clipwise_output']
    # shape (audios_num, classes_num)
    target = output_dict['target'] # shape (audios_num, classes_num)

    cm = metrics.confusion_matrix(np.argmax(target, axis=-1),
                                    np.argmax(clipwise_output, axis=-1),
                                    labels=None)
    val_accuracy = calculate_accuracy(target, clipwise_output)
    print(f', val_acc:{val_accuracy}')
    print('Confusion Matrix:')
    print(cm)
    print(idx_to_lb)

    # Save model
    if iteration % checkpoint_cycle == 0 and iteration > 0:
        checkpoint = {
            'iteration': iteration,
            'model': model.state_dict()}

        checkpoint_name = f'{iteration}_iterations.pth'
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)

        model.save_weights(checkpoint_path)
        print(f'Model weights saved at {checkpoint_name}')
