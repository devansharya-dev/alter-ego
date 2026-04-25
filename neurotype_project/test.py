import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Is GPU Available?", tf.test.is_gpu_available())
print("GPU Device Name:", tf.test.gpu_device_name())

# List all physical devices
devices = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", devices)