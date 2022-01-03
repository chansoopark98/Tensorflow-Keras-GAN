import tensorflow as tf
from model.model import colorization_model, build_discriminator, build_generator

def base_model(image_size, num_classes):

    model_input, model_output = colorization_model(input_shape=(image_size[0], image_size[1], 1), classes=num_classes)
    # final = tf.keras.Model(model_input, model_output)

    # return tf.keras.Model(model_input, model_output)
    return model_input, model_output

def build_gen(image_size, output_channels=3):
    model_input, model_output = build_generator(input_shape=image_size, output_channels=output_channels)

    return model_input, model_output

def build_dis(image_size):
    model_input, model_output = build_discriminator(image_size=image_size)

    return model_input, model_output