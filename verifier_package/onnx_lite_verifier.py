from verifier_package.verifier import Verifier
from converter_package.conversion_format import ConversionFormat
from inference_package.onnx_inference import OnnxInference
from inference_package.lite_inference import TensorflowLiteInference


class OnnxTensorflowLiteVerifier(Verifier):
    def __init__(self):
        super().__init__(ConversionFormat.onnx.value, ConversionFormat.tensorflow_lite.value)

    def verify(self, onnx_model, lite_model, model_name, trace_numpy_input, logger):
        super().verify(onnx_model, lite_model, model_name, trace_numpy_input, logger)

        self.start_timer()
        lite_output = TensorflowLiteInference().run(lite_model, model_name, trace_numpy_input, logger)
        onnx_output = OnnxInference().run(onnx_model, model_name, trace_numpy_input, logger)
        verification_time = self.check_timer()

        max_difference, mismatched_percent = self.get_mismatched(onnx_output, lite_output)

        logger.add(self.get_verification_info(model_name,
                                              verification_time,
                                              max_difference,
                                              mismatched_percent))
