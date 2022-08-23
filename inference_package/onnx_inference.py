import onnxruntime

from inference_package.inference import Inference
from converter_package.conversion_format import ConversionFormat


class OnnxInference(Inference):
    def __init__(self):
        super().__init__(ConversionFormat.onnx.value)

    def run(self, model, model_name, trace_numpy_input, logger):
        super().run(model, model_name, trace_numpy_input, logger)
        self.start_timer()

        session = onnxruntime.InferenceSession(model.SerializeToString())
        session.set_providers(['CPUExecutionProvider'])

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: trace_numpy_input})

        inference_time = self.check_timer()
        logger.add(self.get_inference_info(model_name, inference_time))

        return Inference.adapt_output(output)
