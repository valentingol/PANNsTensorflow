from math import log

import pytest
import tensorflow as tf

from lib.training.metric_loss import cross_entopy_loss, accuracy_metric

@pytest.fixture(scope="module")
def target():
    # targets are integers instead of floats to test type compatibility
    return tf.constant([[0, 0, 0, 0, 0, 1],  # 5
                        [1, 0, 0, 0, 0, 0],  # 0
                        [0, 0, 0, 1, 0, 0],  # 3
                        [0, 0, 1, 0, 0, 0],  # 2
                        [0, 1, 0, 0, 0, 0]]) # 1


@pytest.fixture(scope="module")
def softmax_output():
    return tf.constant([[0.2, 0.1, 0.4, 0.1, 0.1, 0.1],  # wrong
                        [0.4, 0.1, 0.2, 0.1, 0.1, 0.1],  # right
                        [0.2, 0.1, 0.1, 0.4, 0.1, 0.1],  # right
                        [0.3, 0.1, 0.1, 0.1, 0.3, 0.1],  # wrong
                        [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]]) # right


def test_cross_entropy_loss(target, softmax_output):
    # log softmax output
    log_softmax_output = tf.math.log(softmax_output)
    loss = cross_entopy_loss(target, log_softmax_output)

    assert loss.shape == ()
    assert loss.dtype == tf.float32
    expected_loss = - (log(0.1)+log(0.4)+log(0.4)+log(0.1)+log(0.5)) / 30
    assert tf.abs(loss - expected_loss) < 1e-6
    # mix up targets
    lambdas = 0.6
    float_target = tf.cast(target, tf.float32)
    mix_target = float_target * lambdas + float_target[::-1, :] * (1 - lambdas)
    loss = cross_entopy_loss(mix_target, log_softmax_output)

    assert loss.shape == ()
    assert loss.dtype == tf.float32
    expected_loss = 0
    expected_loss -= 0.6 * (log(0.1)+log(0.4)+log(0.4)+log(0.1)+log(0.5)) / 30
    expected_loss -= 0.4 * (log(0.1)+log(0.2)+log(0.4)+log(0.3)+log(0.1)) / 30
    assert tf.abs(loss - expected_loss) < 1e-6


def test_accuracy_metric(target, softmax_output):
    # softmax output
    accuracy = accuracy_metric(target, softmax_output)

    assert accuracy.shape == ()
    assert accuracy.dtype == tf.float32
    assert tf.abs(accuracy - 0.6) < 1e-6
    # log softmax output
    log_softmax_output = tf.math.log(softmax_output)
    accuracy = accuracy_metric(target, log_softmax_output)

    assert accuracy.shape == ()
    assert accuracy.dtype == tf.float32
    assert tf.abs(accuracy - 0.6) < 1e-6
    # mix up targets
    lambdas = 0.6
    float_target = tf.cast(target, tf.float32)
    mix_target = float_target * lambdas + float_target[::-1, :] * (1 - lambdas)
    accuracy = accuracy_metric(mix_target, softmax_output)

    assert accuracy.shape == ()
    assert accuracy.dtype == tf.float32
    assert tf.abs(accuracy - 0.6) < 1e-6
