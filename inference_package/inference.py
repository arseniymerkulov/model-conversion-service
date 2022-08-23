import numpy as np

from inference_package.inference_info import InferenceInfo
from model_process_package.model_process import ModelProcess


class Inference(ModelProcess):
    def __init__(self, input_format):
        super().__init__()
        self.input_format = input_format

    def run(self, model, model_name, trace_numpy_input, logger):
        print(f'run {self.input_format} inference')

    def get_inference_info(self, model_name, inference_time):
        return InferenceInfo(model_name, self.input_format, inference_time).get_json_object()

    @staticmethod
    def adapt_output(output):
        if not isinstance(output, list) and not isinstance(output, tuple):
            output = [output]

        return [np.array(element) for element in output]
