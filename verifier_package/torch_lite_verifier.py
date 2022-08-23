from verifier_package.verifier import Verifier
from converter_package.conversion_format import ConversionFormat
from inference_package.torch_inference import TorchInference
from inference_package.lite_inference import TensorflowLiteInference


class TorchTensorflowLiteVerifier(Verifier):
    def __init__(self):
        super().__init__(ConversionFormat.torch.value, ConversionFormat.tensorflow_lite.value)

    def verify(self, torch_model, lite_model, model_name, trace_numpy_input, logger):
        super().verify(torch_model, lite_model, model_name, trace_numpy_input, logger)

        self.start_timer()
        torch_output = TorchInference().run(torch_model, model_name, trace_numpy_input, logger)
        lite_output = TensorflowLiteInference().run(lite_model, model_name, trace_numpy_input, logger)
        verification_time = self.check_timer()

        max_difference, mismatched_percent = self.get_mismatched(torch_output, lite_output)

        logger.add(self.get_verification_info(model_name,
                                              verification_time,
                                              max_difference,
                                              mismatched_percent))
