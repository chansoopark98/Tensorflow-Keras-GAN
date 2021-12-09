import tensorflow as tf
import tensorflow_io as tfio

"For back up image normalization method"

def rgb2lab(img):
    img_lab = tfio.experimental.color.rgb_to_lab(img)
    L = img_lab[:, :, 0]
    L = (L / 50.) - 1.

    a = img_lab[:, :, 1]
    a = ((a + 127.) / 255.) * 2 - 1.

    b = img_lab[:, :, 2]
    b = ((b + 127.) / 255.) * 2 - 1.

    L = tf.expand_dims(L, -1)
    a = tf.expand_dims(a, -1)
    b = tf.expand_dims(b, -1)
    #
    ab_channel = tf.concat([a, b], axis=-1)

    return L, ab_channel