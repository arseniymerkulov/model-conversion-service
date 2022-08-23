import tensorflow as tf
import numpy as np

from inference_package.inference import Inference
from converter_package.conversion_format import ConversionFormat
from converter_package.quantization_type import QuantizationType
from settings_package.settings import Settings


class TensorflowLiteInference(Inference):
    def __init__(self):
        super().__init__(ConversionFormat.tensorflow_lite.value)

    def run(self, model, model_name, trace_numpy_input, logger):
        super().run(model, model_name, trace_numpy_input, logger)
        self.start_timer()

        # TODO: support different input formats
        if Settings.get_instance().quantization == QuantizationType.full_int.value:
            trace_numpy_input = trace_numpy_input.astype(np.int8)

        model_input = tf.convert_to_tensor(trace_numpy_input)

        interpreter = tf.lite.Interpreter(model_content=model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], model_input)
        interpreter.invoke()

        output = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        inference_time = self.check_timer()
        logger.add(self.get_inference_info(model_name, inference_time))

        return Inference.adapt_output(output)
