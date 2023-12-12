import tensorflow as tf

# Print TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Print Keras version
print(f"Keras Version: {tf.keras.__version__}")

# Check if GPU is available
if tf.test.is_gpu_available():
    # Print CUDA Toolkit version
    print(f"CUDA Toolkit Version: {tf.test.gpu_device_name()}")

    # Print cuDNN version
    print(f"cuDNN Version: {tf.test.is_built_with_cuda()}")

else:
    print("No GPU available. CUDA Toolkit and cuDNN information not applicable.")
