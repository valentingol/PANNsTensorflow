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
            Path leading to data.
        sample_rate : int, optional
            Sample rate used to sample data from audio files. 
            By default 32000.
        max_time : int, optional
            Audio file duration, in seconds. By default 30.
        """
        path = os.path.normpath(path)
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

            if self.file_exists(h5_file_name, path):
                # if h5 file does already exist
                # existing h5 file will be considered
                print("h5 file '", packed_hdf5_path, "' does already exist")

            else:
                # if not, h5 file generation and loading
                print("No existing'", packed_hdf5_path, "' h5 file found")
                print("New '", packed_hdf5_path, "' creation on its way...")
                self._generate_h5_file(path, packed_hdf5_path, max_time, 
                                       sample_rate)

        self._load_h5_file_metadata(packed_hdf5_path)
        print("'", packed_hdf5_path, "' loaded")

        self.packed_hdf5_path = packed_hdf5_path


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


    @staticmethod
    def _get_class_sizes_list(dataset_folder_path :str):
        """Returns the list of number of samples for each class.

        Parameters
        ----------
        dataset_folder_path : str
            path leading to the dataset.

        Returns
        -------
        class_sizes : list
            class_sizes[index] is the number of 
            samples of the class self.index_to_labels[index].
        """
        class_sizes_list=[]
        for class_name in os.listdir(dataset_folder_path):
            class_path = os.path.join(dataset_folder_path, class_name)
            if os.path.isdir(class_path):
                class_sizes_list.append(0)
                for elem in os.listdir(class_path):
                    if os.path.isfile(elem):
                        class_sizes_list[-1] = class_sizes_list[-1]+1

        return class_sizes_list


    def _get_class_sizes_dict(self, class_sizes_list: list):
        """Returns the a dictionnary holding the 
        number of samples for each class.

         Parameters
        ----------
        class_sizes_list : list
            class_sizes_list[index] indicate the number 
            of samples corresponding to index class index.

        Returns
        -------
        class_sizes : dict
            Dictionnary with class labels as key, 
            and the corresponding number of samples as value.
        """
        class_sizes={}
        for index, label in enumerate(self.idx_to_labels):
            class_sizes[label] = class_sizes_list[index]
        return class_sizes


    def get_label_list(self, dataset_folder_path: str):
        """Returns the a dictionnary holding the 
        number of samples for each class.

         Parameters
        ----------
        class_sizes_list : list
            class_sizes_list[index] indicate the number 
            of samples corresponding to index class index.

        Returns
        -------
        class_sizes : dict
            Dictionnary with class labels as key, 
            and the corresponding number of samples as value.
        """
        return [name for name in os.listdir(dataset_folder_path) 
                     if os.path.isdir(os.path.join(dataset_folder_path, name))]


    def _generate_h5_file(self, dataset_folder_path: str, 
                          packed_hdf5_path: str, max_time: int=30, 
                          sample_rate: int=32000):
        """Generate h5 file packing data stored in
        dataset_folder_path.

        Parameters
        ----------
        dataset_folder_path : str
            Path leading to data class folders.
        packed_hdf5_path : str
            Path leading the folder where the h5 file will
            be generate.
        """
        clip_samples = max_time * sample_rate
        # list of class names of the dataset
        # it ignores files in the dataset folder that are not folders
        idx_to_labels = self.get_label_list(dataset_folder_path)

        # mapping class names to class indexes
        labels_to_idx = {
            label: index for index, label in enumerate(idx_to_labels)
        }

        # number of classes of the dataset
        classes_num = len(idx_to_labels)

        # for each class, class_sizes_list[i] is the number samples of the 
        # class with label idx_to_labels[i]
        class_sizes_list = self._get_class_sizes_list(dataset_folder_path)

        print(f"Convert audio files to .h5 in {packed_hdf5_path}   ...")

        # pack dataset in h5 file
        pack_audio_files_to_hdf5(dataset_folder_path, packed_hdf5_path, 
                                 clip_samples, classes_num, 
                                 sample_rate, labels_to_idx, 
                                 idx_to_labels, class_sizes_list)


    def _load_h5_file_metadata(self, packed_hdf5_path: str):
        """Load all metadata 
        (sample_rate, clip_samples, [...]) of an existing h5 dataset.

        Parameters
        ----------
        packed_hdf5_path : str
            Path leading to an h5 file.
        """
        print("h5 file stored in '", packed_hdf5_path, 
              "' is considered for tf.data.Dataset creation")

        self.packed_hdf5_path = packed_hdf5_path
        with h5py.File(packed_hdf5_path, 'r') as f:
            self.sample_rate = f.attrs["sample_rate"]
            self.clip_samples = f.attrs["clip_samples"]
            self.idx_to_labels = f.attrs["labels"]
            self.class_sizes_list = f.attrs["classes_size"]
            self.class_sizes = self._get_class_sizes_dict(self.class_sizes_list)
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
    datasets = AudioDataLoaderH5("datasets/GTZAN_small")
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