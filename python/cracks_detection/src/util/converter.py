import numpy as np
import tensorflow as tf


def convert_model(model_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model1 = converter.convert()
    file = open(r'D:\FAC\surface-crack-detection\model.tflite', 'wb')
    file.write(model1)


def load_model(model_path, dtype):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print("INPUT SHAPE: ", input_shape)
    input_data = np.array(np.random.random_sample(input_shape), dtype=dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_shape = output_details[0]['shape']
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("OUTPUT SHAPE: ", output_shape, "\n")
    # print(output_data)


if __name__ == '__main__':
    convert_model(r"D:\FAC\surface-crack-detection\src\model\unet_crackconcrete_checkpoint.hdf5")
    # load_model(r"D:\FAC\surface-crack-detection\model.tflite", np.float32)
    load_model(r"D:\FAC\surface-crack-detection\mobilenet_v2_deeplab_v3_256_myquant.tflite", np.uint8)
