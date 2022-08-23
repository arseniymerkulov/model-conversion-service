from onnx_tf.backend import prepare
import tensorflow as tf
import shutil
import sys
import numpy as np

from converter_package.converter import Converter
from converter_package.conversion_format import ConversionFormat
from converter_package.quantization_type import QuantizationType
from settings_package.settings import Settings


class OnnxTensorflowLiteConverter(Converter):
    def __init__(self):
        super().__init__(ConversionFormat.onnx.value, ConversionFormat.tensorflow_lite.value)
        self.quantization = Settings.get_instance().quantization

    def set_quantization_settings(self, converter):
        def dynamic_quantization():
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def float16_quantization():
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        def full_integer_quantization():
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 3, 300, 300)
                    yield [data.astype(np.float32)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        {
            QuantizationType.dynamic.value: dynamic_quantization,
            QuantizationType.float16.value: float16_quantization,
            QuantizationType.full_int.value: full_integer_quantization
        }[self.quantization]()

    def convert(self, model, model_name, trace_numpy_input, logger):
        super().convert(model, model_name, trace_numpy_input, logger)
        path = f'{self.output_directory}/{model_name}.pb'

        self.start_timer()
        tf_rep = prepare(model)
        tf_rep.export_graph(path)

        converter = tf.lite.TFLiteConverter.from_saved_model(path)
        self.set_quantization_settings(converter)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # converter.experimental_new_converter = True
        # converter._experimental_lower_tensor_list_ops = False

        model = converter.convert()

        conversion_time = self.check_timer()
        model_size = sys.getsizeof(model)
        logger.add(self.get_conversion_info(model_name, model_size, conversion_time))

        shutil.rmtree(path)

        return model

