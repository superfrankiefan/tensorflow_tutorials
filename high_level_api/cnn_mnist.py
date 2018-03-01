"""
====================================================================================================
Description: Sample for customized estimator. Tensorflow has supplied many predefined estimators,
             such as DNNEstimator, RNNEstimator, et.al But many time, we need our own estimator
             for research or application. Reference this sample we can create our own estimator.
About Tensorflow Estimator:
             Tensorflow Estimator is a high level api for DL, it has supplied many predefined
             structures, such as convolution, pooling, droupout, et.al. We could use Estimator
             to do DL very easily. If we use predefined Estimator, we just need to define input
             function, then it would help us to TRAIN, PREDICT and EVALUATE. And if we want to
             create our own Estimator, we need to define a model_fn which would be used next. And
             it should structure the network and define the output in different mode(TRAIN, EVALUATE,
             PREDICT)
====================================================================================================
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

tf.logging.set_verbosity(tf.logging.INFO)


# define customized estimator
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # 1st layer convolution
    conv1 = tf.layers.conv2d(inputs=input_layer, # define input
                             filters=32, # num of filters
                             kernel_size=[5,5], # filter shape
                             padding="same", # padding to keep the output shape is the same as input
                             activation=tf.nn.relu) # activation function
    # 1st pooling layer: max pooling which select the max num of the pooling area as the output
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2], # pooling filter shape
                                    strides=2) # stride size which means move 2 pixel every move
    # 2nd convolution layer
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)
    # 2nd pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)
    # reshape the pixels into flat
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    # fully connected layer
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024, # num units of last hidden layer
                            activation=tf.nn.relu)
    # regularization using dropout
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4, # dropout rate
                                training=(mode == tf.estimator.ModeKeys.TRAIN))
                                # means this layer would be used when training
    # fully connected layer to calculate the linear combination results,
    # then use this result to calculate softmax probability
    logits = tf.layers.dense(inputs=dropout, units=10)
    # get the predictions
    predictions = {"classes": tf.argmax(input=logits,
                                        axis=1),
                   "probabilities": tf.nn.softmax(logits,
                                                  name="softmax_tensor")}
    # when in predict procedure app would output the predictions
    # EstimatorSpec used to summarize the result
    # Tensorflow has three mode: train, predict, evaluation
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    # define the loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    # When in train procedure, the optimization is needed.
    # And EstimatorSpec used to calculated the training statistics
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # define the evaluation metrics: accuracy
    # when in evaluation mode, the accuracy would be output
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


# define main function
def main(unused_argv):
    # use keras.mnist to load sample mnist data(digital hand writing dataset)
    (x_train, y_train), (x_eval, y_eval) = mnist.load_data()
    train_data = np.asarray(x_train, dtype=np.float16)
    eval_data = np.asarray(x_eval, dtype=np.float16)
    train_labels = np.asarray(y_train, dtype=np.int32)
    eval_labels = np.asarray(y_eval, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, # using our own estimator
                                              model_dir="tmp/mnist_convnet_model")
                                              # define the model dir which used to save the trained model
    # define which information should be logged, and then we could see this using tensorboard
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # define logging hook to clarify how to log information: every 50 epochs write logs
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # define input data for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, # define data
                                                        y=train_labels, # define label
                                                        batch_size=100, # define batch size
                                                        num_epochs=None, # define epoch numbers
                                                        shuffle=True)
    # training model according to train data
    mnist_classifier.train(input_fn=train_input_fn,
                           steps=20000, # define total training steps
                           hooks=[logging_hook])

    # define input data for evaluating
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()