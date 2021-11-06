import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import h5py
import tensorflow as tf

from lib.data.utils.prepare_data import int16_to_float32
from lib.data.utils.prepare_data import pack_audio_files_to_hdf5

class AudioDataLoaderH5:
    """Dataset class for Audio datasets using h5 format."""
    def __init__(self, path: str, sample_rate: int=32000, max_time: int=30):
        """
        Parameters
        ----------
        path : str
            Path leading to the dataset containor. Dataset container is
            either a folder containing subfolders of audio files or a 
            a single .h5 file with good format.
        sample_rate : int, optional
            Sample rate used to sample data from audio files. 
            By default 32000.
        max_time : int, optional
            Audio file duration, in seconds. By default 30.
        """
        path = os.path.normpath(path)
        self.dataset_folder_path = path
        self.sample_rate = sample_rate
        self.max_time = max_time
        # check whether path leads to an h5 file or to the dataset folder
        if os.path.splitext(os.path.split(path)[1])[1] == ".h5":
            # self.path leads to a .h5 file
            print("Loading h5 file '", os.path.split(path)[-1] ,"'.")
            packed_hdf5_path = path
        else:
            # the name of the h5 file is the dataset folder name
            h5_file_name = os.path.split(path)[-1] + ".h5"

            # h5 file will be in the same folder as packed data
            packed_hdf5_path = os.path.join(path, h5_file_name)
            self.packed_hdf5_path = packed_hdf5_path

            if self.file_exists(h5_file_name, path):
                # if h5 file does already exist
                # existing h5 file will be considered
                print("h5 file '", packed_hdf5_path, "' does already exist")

            else:
                # if not, h5 file generation and loading
                print("No existing'", packed_hdf5_path, "' h5 file found")
                print("New '", packed_hdf5_path, "' creation on its way...")
                self._generate_h5_file()

        self._load_h5_file_metadata()
        print("'", packed_hdf5_path, "' loaded")


    @staticmethod
    def file_exists(file_name: str, path :str):
        """Returns True if file_name is in path, False if not.

        Parameters
        ----------
        file_name : str
            name of the file.
        path : str
            Path leading to a file.

        Returns
        -------
            True if file_name is in path, False if not.
        """
        for element_name in os.listdir(path):
            element_path = os.path.join(path, element_name)
            if os.path.isfile(element_path) and element_name == file_name:
                return True
        return False


    def _get_class_sizes_list(self):
        """Returns the list of number of samples for each class.

        Returns
        -------
        class_sizes : list
            class_sizes[index] is the number of 
            samples of the class self.index_to_labels[index].
        """
        class_sizes_list=[]
        for class_name in os.listdir(self.dataset_folder_path):
            class_path = os.path.join(self.dataset_folder_path, class_name)
            if os.path.isdir(class_path):
                class_sizes_list.append(0)
                for elem in os.listdir(class_path):
                    if not(os.path.isfile(elem)) and elem.endswith('.wav'):
                        class_sizes_list[-1] = class_sizes_list[-1]+1

        return class_sizes_list


    def _get_class_sizes_dict(self):
        """Returns the a dictionnary holding the 
        number of samples for each class.

        Returns
        -------
        class_sizes : dict
            Dictionnary with class labels as key, 
            and the corresponding number of samples as value.
        """
        class_sizes={}
        class_sizes_list = self._get_class_sizes_list()
        for index, label in enumerate(self.idx_to_labels):
            class_sizes[label] = class_sizes_list[index]
        return class_sizes


    def get_label_list(self):
        """Returns the a list of labels of the dataset.

        Returns
        -------
        idx_to_labels : list
            Dictionnary with class labels as key, 
            and the corresponding number of samples as value.
        """
        return [name for name in os.listdir(self.dataset_folder_path) 
                     if os.path.isdir(os.path.join(self.dataset_folder_path, name))]


    def _generate_h5_file(self):
        """Generate h5 file packing data stored in
        dataset_folder_path.
        """
        clip_samples = self.max_time * self.sample_rate
        # list of class names of the dataset
        # it ignores files in the dataset folder that are not folders
        idx_to_labels = self.get_label_list()

        # mapping class names to class indexes
        labels_to_idx = {
            label: index for index, label in enumerate(idx_to_labels)
        }

        # number of classes of the dataset
        classes_num = len(idx_to_labels)

        # for each class, class_sizes_list[i] is the number samples of the 
        # class with label idx_to_labels[i]
        class_sizes_list = self._get_class_sizes_list()

        print(f"Convert audio files to .h5 in {self.packed_hdf5_path}   ...")

        # pack dataset in h5 file
        pack_audio_files_to_hdf5(self.dataset_folder_path, 
                                 self.packed_hdf5_path, 
                                 clip_samples, classes_num, 
                                 self.sample_rate, labels_to_idx, 
                                 idx_to_labels, class_sizes_list)
        
        self._load_h5_file_metadata()


    def _load_h5_file_metadata(self):
        """Load all metadata 
        (sample_rate, clip_samples, [...]) of an existing h5 dataset.
        """
        print("h5 file stored in '", self.packed_hdf5_path, 
              "' is considered for tf.data.Dataset creation")

        self.packed_hdf5_path = self.packed_hdf5_path
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            self.sample_rate = f.attrs["sample_rate"]
            self.clip_samples = f.attrs["clip_samples"]
            self.idx_to_labels = f.attrs["labels"]
            self.class_sizes_list = f.attrs["classes_size"]
            self.class_sizes = self._get_class_sizes_dict()
            self.class_size = self.class_sizes_list[0]
            self.classes_num = len(self.idx_to_labels)


    def get_information(self):
        """Print main information about the dataset"""
        print("h5 file where dataset is stored : ", self.packed_hdf5_path)
        print("Sample rate of the dataset : ", self.sample_rate)
        print("Clip samples of the dataset : ", self.clip_samples)
        print("Labels of the dataset : ", self.idx_to_labels)
        print("Number of labels  : ", self.classes_num)
        print("Class sizes : ", self.class_sizes)


    def _train_generator(self):
        """ Generator feeding training dataset."""
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            for i in self.indexes_train:
                audio_name = f["audio_name"][i]
                waveform = int16_to_float32(f["waveform"][i])
                target = f["target"][i]
                yield {
                    "audio_name" : audio_name, 
                    "waveform": waveform, 
                    "target": target
                }


    def _val_generator(self):
        """ Generator feeding validation dataset."""
        with h5py.File(self.packed_hdf5_path, 'r') as f:
            # run through dataset only one time
            for i in self.indexes_val:
                audio_name = f["audio_name"][i]
                waveform = int16_to_float32(f["waveform"][i])
                target = f["target"][i]
                yield {
                    "audio_name" : audio_name, 
                    "waveform": waveform, 
                    "target": target
                }
    ##! NOTE : test_prop argument
    ##! NOTE : test -> val
    def __call__(self, val_prop: float=0.2):
        """Create train and validation datasets.

        Returns
        -------
        train_dataset : tf.data.Dataset
            Dataset containing dictionnaries of format:
            {
                "audio_name": tf.string,
                "waveform": tf.Tensor, shape=(clip_samples,), dtype=tf.float32
                "target": tf.Tensor, shape=(classes_num,), dtype=tf.float32
            }

        val_dataset : tf.data.Dataset
            Dataset containing dictionnary of format:
            {
                "audio_name": tf.string,
                "waveform": tf.Tensor, shape=(clip_samples,), dtype=tf.float32
                "target": tf.Tensor, shape=(classes_num,), dtype=tf.float32
            }
        """
        # creation of a balanced validation and train set
        ##! IDEE : val a la meme prop dans chaque classe que pour le train 
        val_fold = tf.random.uniform((), 0, min(self.class_sizes_list)-1,
                                        dtype=tf.int64)
        with h5py.File(self.packed_hdf5_path, 'r') as hf:
            folds = tf.cast(hf['fold'], tf.int64)

        self.indexes_val = tf.where(folds == val_fold)[:, 0]
        self.indexes_train = tf.where(folds != val_fold)[:, 0]

        # train dataset creation
        train_ds = tf.data.Dataset.from_generator(
            self._train_generator,
            output_signature=({
                "audio_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "waveform": tf.TensorSpec(shape=(self.clip_samples,),
                                          dtype=tf.float32),
                "target": tf.TensorSpec(shape=(self.classes_num,),
                                        dtype=tf.float32)
                })
            )

        # validation dataset creation
        val_ds = tf.data.Dataset.from_generator(
            self._val_generator,
            output_signature=({
                "audio_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "waveform": tf.TensorSpec(shape=(self.clip_samples,),
                                          dtype=tf.float32),
                "target": tf.TensorSpec(shape=(self.classes_num,),
                                        dtype=tf.float32)
                })
            )
        return train_ds, val_ds


if __name__ == "__main__":
    # test the function with GTZAN
    datasets = AudioDataLoaderH5("lib/data/datasets/GTZAN_small")
    train, val = datasets()

    datasets.get_information()

    # print firsts train dataset elements
    for i, data in enumerate(train):
        print(f'Sample {i}, name {data["audio_name"]}, label {data["target"]}.')
        if i >= 50:
            break

    train = train.shuffle(100)
    train = train.batch(2)

    for i, batch in enumerate(train):
        print(f'Batch {i}, names {batch["audio_name"]}, labels {batch["target"]}.')
        if i >= 50:
            break