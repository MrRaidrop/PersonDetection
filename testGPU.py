import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU availability:", tf.config.list_physical_devices('GPU'))

# test your GPU is available for tensorflow