import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import h5py
import tensorflow as tf

from lib.data.utils.prepare_data import int16_to_float32
from lib.data.utils.prepare_data import pack_audio_files_to_hdf5


class AudioDataLoaderH5:
    """Dataset class for Audio datasets using h5 format."""
    def __init__(self, path: str, sample_rate: int=32000, max_time: int=30, 
                 verbose: bool=True):
        """
        Parameters
        ----------
        path : str
            Path leading to the dataset container. Dataset 
            container is either a folder containing subfolders 
            of audio files or a single .h5 file with good format.
            Each subfolder of the dataset container is a class,
            and the class name is the name of the subfolder. Each 
            audio file in the subfolder is a sample of the class.
        sample_rate : int, optional
            Sample rate used to sample data from audio files. 
            By default 32000.
        max_time : int, optional
            Audio file duration, in seconds. By default 30.
        verbose : bool, optional
            If True, print information during dataset loading.
            
        Examples
        --------
        >>> dataset = AudioDataLoaderH5(path='/path/to/dataset/folder')
        or
        >>> dataset = AudioDataLoaderH5(path='/path/to/H5/file.h5')
        
        """
        path = os.path.normpath(path)
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.indexes_train = None
        self.indexes_val = None
        self.dataset_folder_path = None
        self.verbose = verbose

        # check whether path leads to an h5 file or to the dataset folder
        if os.path.isdir(path):
            self.dataset_folder_path = path
            # the name of the h5 file is the dataset folder name
            h5_file_name = os.path.split(path)[-1] + ".h5"

            # h5 file will be in the same folder as packed data
            packed_hdf5_path = os.path.join(path, h5_file_name)
            self.packed_hdf5_path = packed_hdf5_path

            # Warning : if an h5 exists, it will be overwritten
            self._generate_h5_file()

        else:
            # path should leads to a .h5 file
            if self.verbose:
                print("Loading h5 file '", os.path.split(path)[-1] ,"'.")
            packed_hdf5_path = path
            self.packed_hdf5_path = packed_hdf5_path

        # Load metadata from h5 file and store it in self attributes
        self._load_h5_file_metadata()
        if self.verbose:
            print("'", packed_hdf5_path, "' loaded")


    def _get_class_sizes_list(self):
        """Returns the list of number of samples 
        for each class from the dataset folder. 
        This function is relevant only if 
        self.dataset_folder_path is not None, that 
        is to say only if the path provided by the user 
        in self.__init__ lead to the dataset folder, and 
        not to a .h5 file.

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
        """Returns a dictionnary holding the 
        number of samples for each class.
        self.class_sizes_list has to be computed before calling 
        this function.

        Returns
        -------
        class_sizes : dict
            Dictionnary with class labels as key, 
            and the corresponding number of samples as value.
        """
        class_sizes={}
        #Remark : self.class_sizes_list has to be computed before
        class_sizes_list = self.class_sizes_list
        for index, label in enumerate(self.idx_to_labels):
            class_sizes[label] = class_sizes_list[index]
        return class_sizes


    def _get_label_list(self):
        """Returns the a list of labels of the dataset. 
        This function is relevant only if 
        self.dataset_folder_path is not None, that is 
        to say only if the path provided by the user 
        in self__init__ lead to the dataset folder, 
        and not to a .h5 file.

        Returns
        -------
        idx_to_labels : list
            List of labels of the dataset.
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
        idx_to_labels = self._get_label_list()

        # mapping class names to class indexes
        labels_to_idx = {
            label: index for index, label in enumerate(idx_to_labels)
        }

        # number of classes of the dataset
        classes_num = len(idx_to_labels)

        # for each class, class_sizes_list[i] is the number samples of the 
        # class with label idx_to_labels[i]
        class_sizes_list = self._get_class_sizes_list()
        self.class_sizes_list = class_sizes_list
        if self.verbose:
            print(f"Convert audio files to .h5 in {self.packed_hdf5_path}   ...")

        # pack dataset in h5 file
        pack_audio_files_to_hdf5(self.dataset_folder_path, 
                                 self.packed_hdf5_path, 
                                 clip_samples, classes_num, 
                                 self.sample_rate, labels_to_idx, 
                                 idx_to_labels, class_sizes_list)


    def _load_h5_file_metadata(self):
        """Load all metadata 
        (sample_rate, clip_samples, [...]) from the h5 file
        stored at self.packed_hdf5_path.
        """
        if self.verbose:
            print("h5 file stored in '", self.packed_hdf5_path, 
                "' is considered for tf.data.Dataset creation")

        with h5py.File(self.packed_hdf5_path, 'r') as f:
            self.sample_rate = f.attrs["sample_rate"]
            self.clip_samples = f.attrs["clip_samples"]
            self.idx_to_labels = f.attrs["labels"]
            self.class_sizes_list = f.attrs["classes_size"]
            self.class_sizes = self._get_class_sizes_dict()
            self.class_size = self.class_sizes_list[0]
            self.classes_num = len(self.idx_to_labels)
            self.folds = tf.cast(f['fold'], tf.int64)
            self.target = f.attrs['class_index']


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
            # Iterate over all train indexes in the dataset
            # run through dataset only one time
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
            # Iterate over all validation indexes in the dataset
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


    def _get_indexes(self, val_prop: float):
        """ Returns train and validation indexes. 
        For each class with self.class_sizes[i] samples, 
        val_prop proportion of the samples are used for 
        validation. The remaining samples are used for 
        training.

        Parameters
        ----------
        val_prop : float
            Proportion of the dataset to be used for validation.
        
        Returns
        -------
        (indexes_train, indexes_val) : tuple of list
            indexes_train : tf.Tensor with shape (num_train_samples,)
                indexes_train is a list of indexes of the training dataset.
            indexes_val : tf.Tensor with shape (num_val_samples,)
                indexes_val is a list of indexes of the validation dataset.
        """
        indexes_val=[]
        indexes_train=[]
        # Iterate over all classes in the dataset
        for i in range(self.classes_num):
            # Check if the class is not empty
            if self.class_sizes_list[i] != 0:
                # Split the class in train and validation indexes
                mid_i = [int(val_prop * self.class_sizes_list[i])]
                end_i = [self.class_sizes_list[i]-mid_i[0]]

                target = tf.random.shuffle(tf.squeeze(tf.where(self.target == i), axis=[-1]))
                indexes_train.append(tf.slice(target, mid_i, end_i))
                indexes_val.append(tf.slice(target, [0], mid_i))
        
        return tf.concat(indexes_train, axis=0), tf.concat(indexes_val, axis=0)


    def __call__(self, val_prop: float=0.):
        """Create train and validation datasets.
        Parameters
        ----------
        val_prop : float
            Proportion of validation dataset.
        
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
        self.indexes_train, self.indexes_val = self._get_indexes(val_prop)

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
