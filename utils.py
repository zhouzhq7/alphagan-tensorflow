import os
from random import shuffle
import glob
from scipy import misc
import tensorflow as tf
import numpy as np

def load_image(data_dir):
    img_list = glob.glob(data_dir+"/**/*.JPEG")
    save_dir = "./cropped_images_125"
    save_dir_1 = "./cropped_images_150"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_1):
        os.makedirs(save_dir_1)
    t_num_img = len(img_list)
    cnt = 0
    print ("Start...")
    for img in img_list:
        try:
            im = misc.imread(img)
        except:
            continue
        short_edge = min(im.shape[:2])
        long_edge = max(im.shape[:2])

        cnt += 1

        if cnt % 200 == 0:
            print ("{}/{}".format(cnt, t_num_img))

        if short_edge < 120:
            continue
        if long_edge/short_edge < 1.5:
            xx = int((im.shape[1]-short_edge) / 2)
            yy = int((im.shape[0]-short_edge) / 2)
            im = im[yy:yy+short_edge, xx:xx+short_edge]
            im = misc.imresize(im, (224, 224))
            if im.shape != (224, 224, 3):
                continue
            name = img.split('/')[-1]
            misc.imsave(save_dir_1+'/'+name, im)
            if long_edge/short_edge < 1.25:
                misc.imsave(save_dir+'/'+name, im)

def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_test_images(dir='./train_samples'):
    img_paths = glob.glob(dir+'/*.JPEG')
    imgs = []
    for img_path in img_paths:
        im = misc.imread(img_path)
        imgs.append(im)

    return np.array(imgs)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_image_name_to_image_id(name):
    n1, n2 = name.split('_')
    n1 = ''.join(s for s in n1 if s.isdigit())
    n2 = ''.join(s for s in n2 if s.isdigit())

    return np.float32(n1) + np.float32(n2)/1000000

def load_and_save_to_tfrecord(data_dir, save_dir, name):

    mkdir_if_not_exists(save_dir)

    file_name = os.path.join(save_dir, name+'.tf')

    img_paths = glob.glob(data_dir+'/*.JPEG')


    # shuffle all the images
    shuffle(img_paths)

    print ('Writing ', file_name)

    with tf.python_io.TFRecordWriter(file_name) as writer:
        cnt = 0
        for img_path in img_paths:
            cnt += 1
            if cnt % 1000 == 0:
                print ('{}/{}'.format(cnt, len(img_paths)))
            im = misc.imread(img_path)
            img_id = img_path.split('/')[-1]
            img_id = parse_image_name_to_image_id(img_id)
            example = tf.train.Example(
                features = tf.train.Features(
                    feature={'image_raw': _bytes_feature(im.tostring()),
                             'image_id': _float32_feature(img_id)}
                )
            )
            writer.write(example.SerializeToString())

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image.set_shape((256*256*3))

    image = tf.reshape(image, (256, 256, 3))

    return image

def augment(img):
    "j"
    "1. randomly flip the image from left to right"
    img = tf.image.random_flip_left_right(img)

    "2. rotate the image counterclockwise 90 degree"
    img = tf.image.rot90(img, k=1)

    "3. randomly add brightness to image pixels"
    img = tf.image.random_brightness(img, max_delta=63)

    "4. randomly adjust the contrast"
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

    return img

def resize(img):
    img = tf.image.resize_images(img, [64, 64])
    return img

def inputs(filename, batch_size, num_epochs, shuffle_size, is_augment, is_resize):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        if is_augment:
            dataset = dataset.map(augment)

        if is_resize:
            dataset = dataset.map(resize)

        dataset = dataset.shuffle(buffer_size=shuffle_size)

        dataset = dataset.repeat(num_epochs)

        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

#def run():
#    batch_size = 12
#    num_epochs = 12
#
#    filename = './tfrecord_files/images.tf'
#
#    im_batch =  inputs(filename, batch_size, num_epochs, is_augment=False)
#    #init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#    with tf.Session() as sess:
#        #sess.run(init_ops)
#        try:
#            while True:
#                imgs = sess.run(im_batch)
#                print (len(imgs))
#        except tf.errors.OutOfRangeError:
#            print ('Done')

#if __name__ == '__main__':
#    #run()
#    load_and_save_to_tfrecord('../cropped_images_150', './train', 'images150')
