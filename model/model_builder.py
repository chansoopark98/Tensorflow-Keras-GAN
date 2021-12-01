import tensorflow as tf
from model.model import colorization_model

def base_model(image_size, num_classes):

    model_input, model_output = colorization_model(input_shape=(image_size[0], image_size[1], 1), classes=num_classes)
    final = tf.keras.Model(model_input, model_output)

    return final