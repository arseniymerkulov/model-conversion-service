import tensorflow as tf
import sys

from converter_package.converter import Converter
from converter_package.conversion_format import ConversionFormat


class KerasTensorflowLiteConverter(Converter):
    def __init__(self):
        super().__init__(ConversionFormat.keras.value, ConversionFormat.tensorflow_lite.value)

    def convert(self, model, model_name, trace_numpy_input, logger):
        super().convert(model, model_name, trace_numpy_input, logger)

        self.start_timer()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        model = converter.convert()

        conversion_time = self.check_timer()
        model_size = sys.getsizeof(model)
        logger.add(self.get_conversion_info(model_name, model_size, conversion_time))

        return model
