import torch

from inference_package.inference import Inference
from converter_package.conversion_format import ConversionFormat


class TorchInference(Inference):
    def __init__(self):
        super().__init__(ConversionFormat.torch.value)

    def run(self, model, model_name, trace_numpy_input, logger):
        super().run(model, model_name, trace_numpy_input, logger)
        self.start_timer()

        with torch.no_grad():
            model_input = torch.from_numpy(trace_numpy_input)
            model.eval()

            output = model(model_input)

            inference_time = self.check_timer()
            logger.add(self.get_inference_info(model_name, inference_time))

            return Inference.adapt_output(output)
