import tensorflow as tf

def forward_dataset(model: tf.keras.Model, dataset: tf.data.Dataset,
                    return_waveform: bool=False, return_target: bool=False):
    """Forward a full dataset to a model.
    Parameters
    ----------
    model: tf.keras.Model
        The model to forward the dataset to.
    dataset: tf.dataset.Dataset
        The dataset to forward.
    return_waveform: bool, optional
        If True, returns the input waveforms of the model.
        By default False.
    return_target: bool, optional
        If True, returns the target of the imputs.
        By default False.

    Returns:
    --------
    output_dict : dict of tf.Tensor
        Dictionary of tensors containing:
         - 'audio_name': name of inputs, shape=(audios_num,)
         - 'output': output of the model, shape=
           (audios_num, classes_num) if clip-wise,
           (audios_num, segments_num, classes_num) if segment-wise,
           (audios_num, frames_num, classes_num) if frame-wise.
         - 'waveform' (optional): input waveforms,
           shape=(audios_num, segment_samples)
         - 'target' (optional): targets corresponding to each clip,
            shape=(audios_num, classes_num)
    """
    def append_to_dict(dictionary: dict, key: str, value: tf.Tensor):
        """Append a value to the list in a dictionary
        at a given key (create it if not exists)."""
        if key in dictionary.keys():
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    output_dict = {}
    # forward data to a model in batches
    for batch_data_dict in dataset:
        batch_waveform = batch_data_dict['waveform']
        # pass forward in evaluation mode
        batch_output = model(batch_waveform, training=False)

        append_to_dict(output_dict, 'audio_name',
                       batch_data_dict['audio_name'])
        append_to_dict(output_dict, 'output', batch_output)
        if return_waveform:
            append_to_dict(output_dict, 'waveform',
                           batch_data_dict['waveform'])
        if return_target:
            append_to_dict(output_dict, 'target',
                           batch_data_dict['target'])

    for key in output_dict.keys():
        output_dict[key] = tf.concat(output_dict[key], axis=0)

    return output_dict
