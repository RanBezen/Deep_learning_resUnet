import csv
import tensorflow as tf
import os
import glob
import cv2

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimage
# import utils


__author__ = 'assafarbelle'



class CSVSegReaderRandom(object):
    def __init__(self, filenames, image_size=(), crop_size=(128, 128), crops_per_image=50,
                 num_threads=4, capacity=20, min_after_dequeue=10, num_examples=None, data_format='NCHW',
                 random=True):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        raw_filenames = []

        for filename in filenames:
            base_dir = os.path.dirname(filename)
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(os.path.join(base_dir, row[0]) + ':' + os.path.join(base_dir, row[1]))

        self.partial_frame = 0

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            num_examples = min(num_examples, len(raw_filenames))
            raw_filenames = raw_filenames[-num_examples:]
        elif isinstance(num_examples, float) and num_examples < 1:
            self.partial_frame = num_examples
            if num_examples <= 0:
                ValueError('number of examples has to be positive')
            raw_filenames = raw_filenames[-1:]

        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, seed=0, shuffle=random)
        self.image_size = image_size
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.random = random
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.data_format = data_format

    def _get_image(self):

        im_filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        im_filename.set_shape([1, 2])
        im_raw = tf.read_file(im_filename[0][0])
        seg_raw = tf.read_file(im_filename[0][1])

        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')
        if self.partial_frame:
            crop_y_start = int(((1 - self.partial_frame) * self.image_size[0]) / 2)
            crop_y_end = int(((1 + self.partial_frame) * self.image_size[0]) / 2)
            crop_x_start = int(((1 - self.partial_frame) * self.image_size[1]) / 2)
            crop_x_end = int(((1 + self.partial_frame) * self.image_size[1]) / 2)
            image = tf.slice(image, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])
            seg = tf.slice(seg, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])

        return image, seg, im_filename[0][0], im_filename[0][1]

    def get_batch(self, batch_size=1):
        self.batch_size = batch_size
        image_in, seg_in, file_name, seg_filename = self._get_image()
        concat = tf.stack(axis=2, values=[image_in, seg_in])
        image_list = []
        seg_list = []
        filename_list = []
        seg_filename_list = []
        for _ in range(self.crops_per_image):
            cropped = tf.random_crop(concat, [self.crop_size[0], self.crop_size[1], 2])
            shape = cropped.get_shape()
            fliplr = tf.image.random_flip_left_right(cropped)
            flipud = tf.image.random_flip_up_down(fliplr)
            rot_ang = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
            rot = tf.image.rot90(flipud, k=rot_ang)
            rot.set_shape(shape)
            image, seg = tf.unstack(rot, 2, 2)
            image = tf.expand_dims(image, 2)
            seg = tf.expand_dims(seg, 2)
            image_list.append(image)
            seg_list.append(seg)
            filename_list.append(file_name)
            seg_filename_list.append(seg_filename)
        image_many = tf.stack(values=image_list, axis=0)
        seg_many = tf.stack(values=seg_list, axis=0)
        filename_many = tf.stack(values=filename_list, axis=0)
        seg_filename_many = tf.stack(values=seg_filename_list, axis=0)

        (image_batch, seg_batch, filename_batch,
         seg_filename_batch) = tf.train.shuffle_batch([image_many, seg_many, filename_many, seg_filename_many],
                                                      batch_size=self.batch_size,
                                                      num_threads=self.num_threads,
                                                      capacity=self.capacity,
                                                      min_after_dequeue=self.min_after_dequeue,
                                                      enqueue_many=True
                                                      )
        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])

        return image_batch, seg_batch, filename_batch


class CSVSegReader(object):
    def __init__(self, filenames, base_folder='.', image_size=(64, 64, 1), num_threads=4,
                 capacity=20, min_after_dequeue=10, num_examples=None, random=True, data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        num_epochs = None if random else 1
        raw_filenames = []

        for filename in filenames:
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(row[0] + ':' + row[1])

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            raw_filenames = raw_filenames[:num_examples]
            # seg_filenames = seg_filenames[:num_examples]
        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]
            # seg_filenames = [f_name for n, f_name in enumerate(seg_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, num_epochs=num_epochs, shuffle=random, seed=0)
        # self.seg_queue = tf.train.string_input_producer(seg_filenames, num_epochs=num_epochs, shuffle=random, seed=0)

        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.random = random
        self.data_format = data_format

    def _get_image(self):

        filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        filename.set_shape([1, 2])
        # seg_filename = self.seg_queue.dequeue()

        im_raw = tf.read_file(self.base_folder + filename[0][0])
        seg_raw = tf.read_file(self.base_folder + filename[0][1])
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')

        return image, seg, filename[0][0], filename[0][1]

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        image, seg, file_name, seg_file_name = self._get_image()
        if self.random:
            image_batch, seg_batch, filename_batch, seg_filename_batch = tf.train.shuffle_batch(
                [image, seg, file_name, seg_file_name], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.capacity, min_after_dequeue=self.min_after_dequeue)
        else:

            image_batch, seg_batch, filename_batch = tf.train.batch_join([(image, seg, file_name)],
                                                                         batch_size=self.batch_size,
                                                                         capacity=self.capacity,
                                                                         allow_smaller_final_batch=True)

        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])
        return image_batch, seg_batch, filename_batch

def tif2png_dir(data_dir: str, out_dir: str, filename_format='t*.tif'):
    """
    tif2png_dir is a function that converts a directory of tif files to png files
     The inputs to the class are:
        :param data_dir: directory including all image files
        :type data_dir: str
        :param out_dir: directory to output
        :type out_dir: str
        :param filename_format: the format of the files in the directory. use * as a wildcard
        :type filename_format: str

    """

    tif_filenames = glob.glob(os.path.join(data_dir, filename_format))
    tif_filenames.sort()
    os.makedirs(out_dir, exist_ok=True)

    pad_y = 0
    pad_x = 0
    for tif_filename in tif_filenames:
        img = cv2.imread(tif_filename, -1)
        img_size = img.shape
        if img_size[0] % 8:
            pad_y = 8 - (img_size[0] % 8)
        else:
            pad_y = 0
        if img_size[1] % 8:
            pad_x = 8 - (img_size[1] % 8)
        else:
            pad_x = 0
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
        if pad_x or pad_y:
            img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
        base_name = os.path.basename(tif_filename)
        base_name = base_name.replace('.tif', '.png')
        png_filename = os.path.join(out_dir, base_name)
        cv2.imwrite(png_filename, img)
    return pad_y, pad_x
