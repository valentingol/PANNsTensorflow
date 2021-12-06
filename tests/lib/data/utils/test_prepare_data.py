import os
import pytest
import tensorflow as tf

from lib.data.utils.prepare_data import *

def tensor_equal(tensor1, tensor2):
    """Check if two tensors are equal in the sense of sets.
    Examples:
    --------
    >>> tensor_equal(tf.constant([1, 2, 3]), tf.constant([1, 2, 3]))
    True
    >>> tensor_equal(tf.constant([1, 2, 3]), tf.constant([3, 1, 2]))
    False
    >>> tensor_equal(tf.constant([1, 2, 3]), tf.constant([1, 2, 5]))
    False
    """
    tensor1 = tf.cast(tensor1, tf.int64)
    tensor2 = tf.cast(tensor2, tf.int64)
    return bool(
        tf.reduce_prod(
            tf.cast(
                tensor1 == tensor2, tf.int64
                )
            )
        )

def test_float32_to_int16():
    # Test if float32 to int16 conversion works for value greater than 1.0
    assert tensor_equal(float32_to_int16(
            tf.convert_to_tensor([1.5])), tf.convert_to_tensor([2.0**15 - 1])
        )

def test_pad_truncate_sequence():
    # Test if padding works
    assert tensor_equal(pad_truncate_sequence(
            tf.convert_to_tensor([1,2]), 4), tf.convert_to_tensor([1,2,0,0])
        )

    # Test if truncation works
    assert tensor_equal(pad_truncate_sequence(
            tf.convert_to_tensor([1, 2, 3]), 2), tf.convert_to_tensor([1,2])
        )
