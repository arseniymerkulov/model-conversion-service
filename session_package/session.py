import requests


from extractor_package.torch_extractor import TorchExtractor
from extractor_package.keras_extractor import KerasExtractor

from converter_package.conversion_format import ConversionFormat
from converter_package.torch_onnx_converter import TorchOnnxConverter
from converter_package.onnx_lite_converter import OnnxTensorflowLiteConverter
from converter_package.keras_lite_converter import KerasTensorflowLiteConverter

from verifier_package.torch_onnx_verifier import TorchOnnxVerifier
from verifier_package.torch_lite_verifier import TorchTensorflowLiteVerifier
from verifier_package.keras_lite_verifier import KerasTensorflowLiteVerifier

from saver_package.lite_saver import TensorflowLiteSaver

from pipeline_package.pipeline import Pipeline
from pipeline_package.pipeline_type import PipelineType

from settings_package.settings import Settings


class Session:
    def __init__(self):
        settings = Settings.get_instance()

        self.authorization = settings.authorization
        self.allowed_extensions = settings.allowed_extensions
        self.rest_endpoint_url = settings.rest_endpoint_url

        self.pipeline = Pipeline()
        {
            PipelineType.torch_to_tensorflow_lite.value: self.set_torch_to_lite_pipeline,
            PipelineType.keras_to_tensorflow_lite.value: self.set_keras_to_lite_pipeline
        }[settings.pipeline_type]()

    def check_file_extension(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def check_token(self, token):
        headers = {'Authorization': token}

        return requests.get(
            self.rest_endpoint_url + '/api/check',
            headers=headers,
            verify=True
        ).ok

    def set_torch_to_lite_pipeline(self):
        self.pipeline.add_extractor(TorchExtractor())

        self.pipeline.add_converter(TorchOnnxConverter())
        self.pipeline.add_converter(OnnxTensorflowLiteConverter())

        self.pipeline.add_verifier(TorchOnnxVerifier())
        self.pipeline.add_verifier(TorchTensorflowLiteVerifier())

        self.pipeline.add_saver(TensorflowLiteSaver())

    def set_keras_to_lite_pipeline(self):
        self.pipeline.add_extractor(KerasExtractor())

        self.pipeline.add_converter(KerasTensorflowLiteConverter())
        self.pipeline.add_verifier(KerasTensorflowLiteVerifier())

        self.pipeline.add_saver(TensorflowLiteSaver())

    def get_model_path(self, model_filename):
        return self.pipeline.get_model_path(model_filename)

    def upload_preprocessing(self):
        self.pipeline.clear_output_archive()

    def upload_postprocessing(self, model_filename):
        return self.pipeline.run_model(model_filename)
