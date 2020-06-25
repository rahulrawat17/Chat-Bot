import tensorflow as tf
graph_def_file = "frozen_model_Sample_model.pb"
input_arrays = ['ei_0','ei_1','ei_2','ei_3','ei_4','ei_5','ei_6','ei_7','ei_8','ei_9','ei_10','ei_11','ei_12','ei_13','ei_14','ei_15','ei_16','ei_17','ei_18','ei_19']
output_arrays = ["decoder/rdx_output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays, input_shapes = [1,20,1])
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
