import pytest
import tensorflow as tf

from lib.data.datasets import AudioDataLoaderH5

@pytest.fixture(scope="module")
def dataset_with_h5():
    dataset_with_h5 = AudioDataLoaderH5("tests/lib/data/datasets/GTZAN_small_with_h5")
    return dataset_with_h5

## MultiHeadAttention class
def test_init(dataset_with_h5):
    assert dataset_with_h5.packed_hdf5_path == "tests/lib/data/datasets/GTZAN_small_with_h5/GTZAN_small_with_h5.h5"
    

def test_get_label_list(dataset_with_h5):
    assert dataset_with_h5.get_label_list("tests/lib/data/datasets/GTZAN_small_with_h5") == ["blues", 
                                                               "classical", 
                                                                "country"]

def test__get_class_sizes_list(dataset_with_h5):
    assert dataset_with_h5._get_class_sizes_list("tests/lib/data/datasets/GTZAN_small_with_h5") == [3, 3, 3]

def test__get_class_sizes_dict(dataset_with_h5):
    pass