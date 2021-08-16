import tensorflow as tf


def cross_entopy_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Perform cross entropy loss (one-hot encoded targets).

    Parameters
    ----------
    y_true : tf.Tensor
        Contains targets that are one-hot encoded.
        The function supports mixup in targets.
    y_pred : tf.Tensor
        Contains model outputs that are log softax outputs.

    Returns
    -------
    loss: scalar tf.Tensor
        Mean of cross entropy loss across all inputs.
    """
    # ensure that y_true and y_pred has the good type
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    loss = - y_true * y_pred
    loss = tf.reduce_mean(loss)
    return loss


def accuracy_metric(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Calculate accuracy

    Parameters
    ----------
    y_true : tf.Tensor
        Contains targets that are one-hot encoded.
        The function supports mixup in targets.
    y_pred : np.array
        Contains model outputs (can be softmax outputs
        or log softmax outputs).

    Returns
    -------
    accuracy : scalar tf.Tensor
        Accuracy of the predictions by comparison
        with targets (using argmax).
    """
    good_preds = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1))
    good_preds = tf.cast(good_preds, tf.float32)
    accuracy = tf.reduce_mean(good_preds)
    return accuracy
