from verifier_package.verifier import Verifier
from converter_package.conversion_format import ConversionFormat
from inference_package.keras_inference import KerasInference
from inference_package.lite_inference import TensorflowLiteInference


class KerasTensorflowLiteVerifier(Verifier):
    def __init__(self):
        super().__init__(ConversionFormat.keras.value, ConversionFormat.tensorflow_lite.value)

    def verify(self, keras_model, lite_model, model_name, trace_numpy_input, logger):
        super().verify(keras_model, lite_model, model_name, trace_numpy_input, logger)

        self.start_timer()
        keras_output = KerasInference().run(keras_model, model_name, trace_numpy_input, logger)
        lite_output = TensorflowLiteInference().run(lite_model, model_name, trace_numpy_input, logger)
        verification_time = self.check_timer()

        max_difference, mismatched_percent = self.get_mismatched(keras_output, lite_output)

        logger.add(self.get_verification_info(model_name,
                                              verification_time,
                                              max_difference,
                                              mismatched_percent))
