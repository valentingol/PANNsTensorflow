import os

import numpy as np
import tensorflow as tf
from sklearn import metrics

import h5py
import os, time
import librosa
from tqdm import tqdm

augmentation = ['mixups']


##############################################################################
########################### MIXUP #####################################

##############################################################################

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, 
                                         self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out




##############################################################################
########################### DATASET #####################################

##############################################################################

class AudioDataset:
    """Represent a dataset of audio samples
    """
    def __init__(self, batch_size=8, holdout_fold = 2, shuffle=True, 
                 sample_rate=32000, max_time=30, path=r'datasets\genres', 
                 loading_format=".h5py", h5py_path=None, augmentation=["mixup"]
                 ,labels = ['blues', 'classical', 'country', 'disco','hiphop', 
                            'jazz', 'metal', 'pop', 'reggae', 'rock']):
        """Initialize of dataset utils

        Args:
            batch_size (int, optional): Size of each batch of datasets. 
                                        Defaults to 8.
            holdout_fold (int, optional): Validation data fold. Defaults to 2.
            shuffle (bool, optional): Shuffle dataset if True, do nothing else. 
                                       Defaults to True.
            sample_rate (int, optional): Sample rate of inputs. 
                                         Defaults to 32000.
            max_time (int, optional): Time of waveform sequence. If sequence is
                                      not max_time lenght, we cut it, or pad it. 
                                      Defaults to 30.
            path (regexp, optional): Path of data folder of GTZAN. 
                                     Defaults to r'datasets\genres'.
            loading_format (str, optional): Data loading format. 
                                            Defaults to ".h5py".
            h5py_path ([type], optional): Path of h5 file if it exist. 
                                          If not None, it load the h5 file 
                                          specisfied by h5py_path. 
                                          Defaults to None.
            augmentation (list, optional): Augmentation info. If "mixup" in 
                                           augmentation,train_batches size is 
                                           multiply by 2.
                                           Defaults to ["mixup"].
            labels (list, optional): Labels of dataset. 
                                     Defaults to ['blues', 'classical', 
                                                'country', 'disco','hiphop', 
                                                'jazz', 'metal', 'pop', 
                                                'reggae', 'rock'].
        """
        if 'mixup' in augmentation:
            self.batch_size = 2*batch_size
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

        print("You choose '" + loading_format + "' as loading format")

        if loading_format == ".h5py" and self.packed_hdf5_path is None:
            self.packed_hdf5_path = os.path.join(self.path, "GTZAN_dataset.h5")
            print("Convert audio files to '" + loading_format + "' ...")
            print(self.packed_hdf5_path)
            self.pack_audio_files_to_hdf5()

    @staticmethod
    def traverse_folder(fd):
        paths = []
        names = []
        for root, dirs, files in os.walk(fd):
            for name in files:
                if name.endswith('.wav'):
                    filepath = os.path.join(root, name)
                    names.append(name)
                    paths.append(filepath)
        return names, paths
    
    @staticmethod
    def float32_to_int16(x):
        if np.max(np.abs(x)) > 1.:
            x /= np.max(np.abs(x))
        return (x * 32767.).astype(np.int16)
    
    def to_one_hot(self, k):
        target = np.zeros(self.classes_num)
        target[k] = 1
        return target
    
    def pad_truncate_sequence(self, x):
        if len(x) < self.clip_samples:
            return np.concatenate((x, np.zeros(self.clip_samples - len(x))))
        else:
            return x[0:self.clip_samples]
        
    def get_target(self, audio_name):
        return self.lb_to_idx[audio_name.split('.')[0]]
    
    def pack_audio_files_to_hdf5(self):
        # Paths
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
            'target': tf.convert_to_tensor(np.array(targets)), 
            'fold': tf.convert_to_tensor(np.arange(len(audio_names)) % 10 + 1)}

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
    
            for n in tqdm(range(audios_num), total=audios_num):
                audio_name = meta_dict['audio_name'][n]
                fold = meta_dict['fold'][n]
                audio_path = meta_dict['audio_path'][n]
                target = meta_dict['target'][n]

                (audio, _) = librosa.core.load(audio_path, sr=self.sample_rate, 
                                               mono=True)
                audio = self.pad_truncate_sequence(audio)

                hf['audio_name'][n] = audio_name.encode()
                hf['waveform'][n] = tf.convert_to_tensor(
                                                self.float32_to_int16(audio)
                                                )
                hf['target'][n] = self.to_one_hot(target)
                hf['fold'][n] = fold

        print('Write hdf5 to {}'.format(self.packed_hdf5_path))
        print('Time: {:.3f} s'.format(time.time() - feature_time))

    @staticmethod
    def int16_to_float32(x):
        return (x / 32767.).astype(np.float32)
    
    def train_generator(self):
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            while True:
                for i in self.indexes_train:
                    yield {"audio_name":f["audio_name"][i], 
                           "waveform":tf.expand_dims(
                                        self.int16_to_float32(f["waveform"][i])
                                        , axis=-1), 
                           "target":f["target"][i]}
                if self.shuffle:
                    tf.random.set_seed(self.random_seed)
                    self.indexes_train = tf.random.shuffle(self.indexes_train)

    def val_generator(self):
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            for i in self.indexes_val:
               yield {"audio_name":f["audio_name"][i], 
                      "waveform":tf.expand_dims(
                                    self.int16_to_float32(f["waveform"][i]), 
                                    axis=-1),
                      "target":f["target"][i]}

    def __call__(self, random_seed = 1234):
        """Create batched train and validation datasets

        Args:
            random_seed (int, optional): Random seed. Defaults to 1234.

        Returns:
            train_loader (tf.data.Dataset): 
            
            dataset of dictionnary of shape 
            {"audio_names":(batch_size,),
             "waveform": (batch_size, self.clip_samples, 1),
             "target": (batch_size, self.classes_num)
            
            val_loader (tf.data.Dataset): 
            
            dataset of dictionnary of shape 
            {"audio_names":(batch_size,),
             "waveform": (batch_size, self.clip_samples, 1),
             "target": (batch_size, self.classes_num)
        """
        self.random_seed = random_seed
        
        with h5py.File(self.packed_hdf5_path, 'r') as hf:
            folds = tf.cast(hf['fold'][:], tf.int64)
        self.holdout_fold = tf.cast(self.holdout_fold, tf.int64)
        self.indexes_train = tf.where(folds != self.holdout_fold)[:,0]
        self.indexes_val = tf.where(folds == self.holdout_fold)[:,0]
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

        return train_ds.cache().batch(self.batch_size), val_ds.cache().batch(self.batch_size)

dataset=AudioDataset(augmentation=augmentation, h5py_path=r"C:\Users\ubar\Documents\ML_Projects\PANNsTensorflow\datasets\genres\GTZAN_dataset.h5")
train_loader, validate_loader = dataset()
idx_to_lb = dataset.idx_to_lb

##############################################################################
########################### REQUIREMENTS #####################################
# model: tf.keras.Model
# train_loader, validate_loader: tf.dataset.Dataset of dict (keys: 'waveform',
# 'target', 'audio_name', 'mixup_lambda')
# augmentation: dict for augmentation
# Mixup: mix-up class
# do_mixup: apply mix-up on targets
# idx_to_lb: dict for label index to class name
##############################################################################

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
model.complile(optimizer=optimizer)
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
from sklearn import metrics

init_iteration = 0
stop_iteration = 1000
validation_cycle = 50
checkpoint_cycle = 50

# Train on mini batches
for iteration in range(init_iteration, stop_iteration+1):
    for batch_data_dict in train_loader:

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

        # Validation
        if iteration % validation_cycle == 0 and iteration > 0:
            output_dict = forward(model, validate_loader, return_target=True)
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

            model.save(checkpoint_path)
            print(f'Model saved at {checkpoint_name}')

        print()




if __name__ == "__main__":
    #Create the dataset with default params
    dataset=AudioDataset(augmentation=augmentation, h5py_path=r"C:\Users\ubar\Documents\ML_Projects\PANNsTensorflow\datasets\genres\GTZAN_dataset.h5")
    train_loader, validate_loader = dataset()
    
    #Print the 10 first batchs of train_loader
    stop_iter = 10
    for i, batch in enumerate(train_loader):
        print(batch)
        if i == stop_iter:
            break
    print("________________________________________________")  
    print("________________________________________________")
    #Print all batches of val_loader
    for batch in validate_loader:
        print(batch)