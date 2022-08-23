import numpy as np


from pipeline_package.pipeline_type import PipelineType
from converter_package.quantization_type import QuantizationType


class Settings:
    instance = None

    def __init__(self):
        self.input_size = 300
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.input_type = np.float32
        self.threshold = 1e-02

        self.images_directory = 'images'
        self.output_directory = 'output'

        self.pipeline_type = PipelineType.torch_to_tensorflow_lite.value
        self.quantization = QuantizationType.float16.value

        self.authorization = False
        self.allowed_extensions = ['mar', 'zip']
        self.rest_endpoint_url = 'https://skuvision.edgesoft.ru:9060'

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = Settings()

        return cls.instance
