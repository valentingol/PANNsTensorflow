import os
import pytest
import tensorflow as tf

from lib.data.dataset import AudioDataLoaderH5


good_h5_pth = os.path.normpath("tests/datasets/GTZAN_small/GTZAN_small.h5")

@pytest.fixture(scope="module")
def dataset():
    #Generate a dataset object from the GTZAN_small dataset folder
    dataset = AudioDataLoaderH5("tests/datasets/GTZAN_small", 
                                max_time=1)
    dataset.get_information()
    yield dataset
    #remove the h5 file to recover the original dataset folder
    os.remove(dataset.packed_hdf5_path)

def test_init(dataset):
    #Check whether h5 packed audio files are stored in the right place
    assert dataset.packed_hdf5_path == good_h5_pth
    
    #Check whether the h5 file is loaded correctly
    dataset = AudioDataLoaderH5(good_h5_pth)
    

def test_get_label_list(dataset):
    assert dataset._get_label_list() == ["blues", 
                                        "classical", 
                                        "country",
                                        "rock"
                                        ]

def test__get_class_sizes_list(dataset):
    assert dataset._get_class_sizes_list() == [0, 2, 1, 5]

def test__get_class_sizes_dict(dataset):
    dataset._get_class_sizes_dict() == {'blues': 0, 
                                        'classical': 2, 
                                        'country': 1,
                                        'rock': 5
                                        }

def test__load_h5_file_metadata(dataset):
    dataset._load_h5_file_metadata()
    assert dataset.sample_rate == 32000
    assert dataset.clip_samples == 32000
    assert dataset.idx_to_labels == ["blues", 
                                     "classical", 
                                     "country",
                                     "rock"
                                     ]
    assert dataset.class_sizes_list == [0, 2, 1, 5]
    assert dataset.class_sizes == {'blues': 0, 
                                  'classical': 2, 
                                  'country': 1,
                                  'rock': 5
                                }
    assert dataset.class_size == 0
    assert dataset.classes_num == 4
    #assert dataset.target == 0 #TODO : rename target

def test__get_indexes(dataset):
    #Test the function for a validation proportion of 0.25
    train_indexes, val_indexes = dataset._get_indexes(0.25)
    train_indexes = set(train_indexes.numpy())
    val_indexes = list(val_indexes.numpy())
    assert train_indexes == set([0, 1, 2, 3, 4, 5, 6]) and val_indexes == [7] \
           or \
           train_indexes == set([0, 1, 2, 3, 4, 5, 7]) and val_indexes == [6] \
           or \
           train_indexes == set([0, 1, 2, 3, 4, 6, 7]) and val_indexes == [5] \
           or \
           train_indexes == set([0, 1, 2, 3, 5, 6, 7]) and val_indexes == [4] \
           or \
           train_indexes == set([0, 1, 2, 4, 5, 6, 7]) and val_indexes == [3]
     
    #Test the function for a validation proportion of 0.0 
    train_indexes, val_indexes = dataset._get_indexes(0.)
    train_indexes = set(train_indexes.numpy())
    val_indexes = list(val_indexes.numpy())
    assert train_indexes == set([0, 1, 2, 3, 4, 5, 6, 7]) and val_indexes == []

    #Test the function for a validation proportion of 1.0
    train_indexes, val_indexes = dataset._get_indexes(1.)
    train_indexes = list(train_indexes.numpy())
    val_indexes = set(val_indexes.numpy())
    assert train_indexes == [] and val_indexes == set([0, 1, 2, 3, 4, 5, 6, 7])

def test__train_generator(dataset):
    labels_to_idx = {'b\'blues':0,
                     'b\'classical':1, 
                     'b\'country':2, 
                     'b\'rock':3}

    dataset.indexes_train, dataset.indexes_val = dataset._get_indexes(0.0)
    for sample in dataset._train_generator():
        assert labels_to_idx[str(sample["audio_name"]).split(".")[0]] \
            == tf.argmax(sample["target"])

def test__val_generator(dataset):
    labels_to_idx = {'b\'blues':0,
                     'b\'classical':1, 
                     'b\'country':2, 
                     'b\'rock':3}

    dataset.indexes_train, dataset.indexes_val = dataset._get_indexes(1.)
    for sample in dataset._val_generator():
        assert labels_to_idx[str(sample["audio_name"]).split(".")[0]] \
            == tf.argmax(sample["target"])

def test___call__(dataset):
    train_ds, val_ds = dataset(0.25)
    assert len(list(train_ds)) == 7 and len(list(val_ds)) == 1
    
    labels_to_idx = {'b\'blues':0,
                     'b\'classical':1, 
                     'b\'country':2, 
                     'b\'rock':3}
    
    for sample in iter(train_ds):
        assert labels_to_idx[str(sample["audio_name"].numpy()).split(".")[0]] \
            == tf.argmax(sample["target"])

    for sample in iter(val_ds):
        assert labels_to_idx[str(sample["audio_name"].numpy()).split(".")[0]] \
            == tf.argmax(sample["target"])
