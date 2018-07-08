import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import cv2
from Network import Network
from test_Params import TestParams
import re


import zipfile


def zipdir(path, zipfilename):
    with zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), arcname=file)


def log_print(string):
    now_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S: ')
    print(now_string + string)


def get_test_filenames(data_dir, filename_regexp):
    dir_files = os.listdir(data_dir)
    regex = re.compile(filename_regexp)

    selected_files = [os.path.join(data_dir, l) for l in filter(regex.search, dir_files)]
    return selected_files


def convert_net_output_to_seg(net_output):
    hard_seg = tf.arg_max(net_output, -1, tf.int32, 'hard_segmentation')
    return hard_seg


def test():
    params = TestParams()
    # Data input
    with tf.device('/cpu:0'):
        with tf.name_scope('Data'):
            test_image_pl = tf.placeholder(dtype=tf.float32, shape=params.image_shape, name='input_image_placeholder')
            test_image = tf.reshape(test_image_pl, [1]+ params.image_shape + [1])
    # Build Network Graph

    net = Network()
    device = '/gpu:0' if params.use_gpu else '/cpu:0'
    with tf.device(device):
        with tf.name_scope('val_tower'):
            with tf.variable_scope('net'):
                log_print('Building validation network')
                norm_val_image_batch = tf.div(tf.subtract(test_image, params.norm),
                                              params.norm)
                net_segs_soft = net.build(norm_val_image_batch, **params.net_build_params)
                net_segs = convert_net_output_to_seg(net_segs_soft)
                log_print('Done building validation network')

    output_dir = params.experiment_log_dir
    saver = tf.train.Saver(var_list=tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        log_print('Started TF Session')
        sess.run(init_op)
        log_print('Variables Initialize')
        saver.restore(sess, params.load_checkpoint_path)
        log_print('Params loaded from checkpoint: {}'.format(params.load_checkpoint_path))
        threads = tf.train.start_queue_runners(sess, coord)
        log_print('Started {} data threads'.format(len(threads)))
        elapsed_time = 0.
        end_time = 0.
        other_time = 0.
        options = tf.RunOptions()
        run_metadata = tf.RunMetadata()
        feed_dict = {}
        file_list = get_test_filenames(params.data_dir, params.filename_regexp)
        for image_filename in file_list:
            try:
                start_time = time.time()
                other_time += start_time - end_time
                img = cv2.imread(image_filename, -1)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(image_filename))
                feed_dict[test_image_pl] = img
                seg_image = sess.run(
                    net_segs,
                    feed_dict=feed_dict,
                    options=options,
                    run_metadata=run_metadata)
                end_time = time.time()
                elapsed_time += end_time - start_time
                log_print('Done Segmentation for file: {}'.format(image_filename))
                out_filename = os.path.join(output_dir, 'seg_{}'.format(os.path.basename(image_filename)))
                out_shape = seg_image.shape
                if len(out_shape) > 2 and out_shape[0] == 1:
                    seg_image = np.squeeze(seg_image)

                success = cv2.imwrite(out_filename, seg_image.astype(np.uint8))
                if success:
                    log_print('Saved Segmentation to file: {}'.format(out_filename))
                else:
                    raise ValueError('Could not save file {} for input image {}'.format(out_filename, image_filename))

            except (RuntimeError, KeyboardInterrupt):

                return

    log_print('Creating zip file')
    zipdir(path=output_dir, zipfilename=os.path.join(output_dir, '..', 'Outputs.zip'))
    log_print('Done!')


if __name__ == '__main__':
    params = TestParams()

    test()
