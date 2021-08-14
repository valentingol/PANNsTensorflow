import pytest
import tensorflow as tf

from lib.training.train_utils import forward_dataset

@pytest.fixture(scope="module")
def model():
    kl = tf.keras.layers
    # log softmax outputs
    return tf.keras.Sequential([
               kl.Flatten(),
               kl.Dense(64, activation='relu'),
               kl.Dense(3, activation='softmax'),
               kl.Lambda(lambda x: tf.math.log(x)),
               ])

@pytest.fixture(scope="module")
def dataset():
    # names are composed of 10 random characters for each sample
    name_int = tf.random.uniform((16*10, ), 32, 123, dtype=tf.int64)
    name = tf.map_fn(lambda x: chr(x), name_int, fn_output_signature=tf.string)
    name = tf.strings.reduce_join(tf.reshape(name, (16, 10)), axis=1)

    waveform = tf.random.normal((16, 128), dtype=tf.float32)
    target_int = tf.random.uniform((16, ), 0, 3, dtype=tf.int64)
    target = tf.one_hot(target_int, 3, dtype=tf.int64)
    data_dict = {'audio_name': name, 'waveform': waveform, 'target': target}
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    return dataset


def test_forward_dataset(model, dataset):
    for full_data_dic in dataset.batch(len(dataset)).take(1):
        name_ds = full_data_dic['audio_name']
        waveform_ds = full_data_dic['waveform']
        target_ds = full_data_dic['target']
    batch_dataset = dataset.batch(3)
    # return_waveform, return_target = False, False
    output_dict = forward_dataset(model, batch_dataset, return_waveform=False,
                                  return_target=False)
    assert len(output_dict) == 2
    assert {'audio_name', 'output'} <= output_dict.keys()
    name, output = output_dict['audio_name'], output_dict['output']
    assert tf.reduce_all(tf.equal(name, name_ds))
    assert output.dtype is tf.float32
    assert output.shape == (16, 3)
    # return_waveform, return_target = True, False
    output_dict = forward_dataset(model, batch_dataset, return_waveform=True,
                                  return_target=False)
    assert len(output_dict) == 3
    assert {'audio_name', 'output', 'waveform'} <= output_dict.keys()
    waveform = output_dict['waveform']
    assert tf.reduce_all(tf.equal(waveform, waveform_ds))
    # return_waveform, return_target = False, True
    output_dict = forward_dataset(model, batch_dataset, return_waveform=False,
                                  return_target=True)
    assert len(output_dict) == 3
    assert {'audio_name', 'output', 'target'} <= output_dict.keys()
    target = output_dict['target']
    assert tf.reduce_all(tf.equal(target, target_ds))
    # return_waveform, return_target = True, True
    output_dict = forward_dataset(model, batch_dataset, return_waveform=True,
                                  return_target=True)
    assert len(output_dict) == 4
    assert {'audio_name', 'output', 'waveform', 'target'} <= output_dict.keys()
