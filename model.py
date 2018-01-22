import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn.externals import joblib


def warp(pg, pts, p_w=400, p_h=600):
    d_pts = np.array(((0, 0), (p_w, 0), (p_w, p_h), (0, p_h)), np.float32)
    s_pts = np.reshape(pts, (4, 2))
    M = cv2.getPerspectiveTransform(s_pts, d_pts)
    return cv2.warpPerspective(pg, M, (p_w, p_h))

# cnn architecture
def conv2d(inputs, filters):
    layer = tf.layers.conv2d(inputs=inputs,
                             filters=filters,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             padding='valid',
                             activation=tf.nn.relu)
    layer = tf.layers.max_pooling2d(layer, pool_size=[2, 2], strides=2)
    return layer

def build_cnn(inputs):
    layer1 = conv2d(inputs=inputs, filters=8)
    layer2 = conv2d(inputs=layer1, filters=16)
    layer3 = conv2d(inputs=layer2, filters=32)
    layer4 = conv2d(inputs=layer3, filters=64)
    flat = tf.contrib.layers.flatten(layer4)
    logits = tf.layers.dense(inputs=flat, units=8)
    return logits


def fix(image_path):
    # load scaler
    scaler = joblib.load('scaler.save')
    # read image
    image_to_fix = mpimg.imread(image_path)
    w, h, ch = image_to_fix.shape
    # tensorflow graph placeholders
    images_ph = tf.placeholder(tf.float32, (None, w // 2, h // 2, ch), name='images')
    # build network
    outs = build_cnn(images_ph)
    # predict pts
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph('./tf_model/model.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./tf_model'))
        # resize image
        test_img = cv2.resize(image_to_fix, (image_to_fix.shape[1]//2, image_to_fix.shape[0]//2))
        img = np.asarray(test_img)
        prediction = sess.run(outs, feed_dict={images_ph: img[None, :, :, :]})
        # inverse scale
        pts = scaler.inverse_transform(prediction)
        # warp
        image = warp(image_to_fix, pts)
        cv2.imshow("fixed", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Restore Perspective')
    parser.add_argument('image_path',
                        type=str,
                        help='Provide a path to image that you want to process')
    args = parser.parse_args()
    fix(args.image_path)

