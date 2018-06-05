import tensorflow as tf
from unet import unet


class Network(object):
    def __init__(self):
        pass

    def build(self, input_batch, example_parameter_string='Example', example_parameter_int=3):
        """
        This function is where you write the code for your network.
          The input is a batch of images of size (N,H,W,1)
        N is the batch size
        H is the image height
        W is the image width
        The output needs to be of shape (N,H,W,3)
        where the last channel is the UNNORMALIZEd calss probabilities
          (before softmax) for classes background,
        foreground and edge.
        :param input_batch:
        :param example_parameter_int:
        :param example_parameter_string: A parameter for example.
                                           See Params file to change the value
        :return:
        """
        for i in range(example_parameter_int):
            print('{}: '.format(i) + example_parameter_string)
        # w = tf.get_variable(name='w', shape=(1, 1, 1, 3), dtype=tf.float32,
        #                     initializer=tf.random_normal_initializer)
        # out = tf.nn.conv2d(input_batch, w, strides=[1, 1, 1, 1], padding='SAME')
        out = unet(input_batch)
        tf.summary.scalar('my_summary', tf.reduce_mean(out))
        return out