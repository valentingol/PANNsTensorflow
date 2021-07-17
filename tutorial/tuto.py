import os

import numpy as np
import tensorflow as tf
from sklearn import metrics

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
model.complile(optimizer=optimizer))
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
