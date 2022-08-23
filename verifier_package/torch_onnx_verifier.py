from verifier_package.verifier import Verifier
from converter_package.conversion_format import ConversionFormat
from inference_package.onnx_inference import OnnxInference
from inference_package.torch_inference import TorchInference


class TorchOnnxVerifier(Verifier):
    def __init__(self):
        super().__init__(ConversionFormat.torch.value, ConversionFormat.onnx.value)

    def verify(self, torch_model, onnx_model, model_name, trace_numpy_input, logger):
        super().verify(torch_model, onnx_model, model_name, trace_numpy_input, logger)

        self.start_timer()
        torch_output = TorchInference().run(torch_model, model_name, trace_numpy_input, logger)
        onnx_output = OnnxInference().run(onnx_model, model_name, trace_numpy_input, logger)
        verification_time = self.check_timer()

        max_difference, mismatched_percent = self.get_mismatched(torch_output, onnx_output)

        logger.add(self.get_verification_info(model_name,
                                              verification_time,
                                              max_difference,
                                              mismatched_percent))
