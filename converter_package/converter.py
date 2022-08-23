from model_process_package.model_process import ModelProcess
from converter_package.conversion_info import ConversionInfo

from settings_package.settings import Settings


class Converter(ModelProcess):
    def __init__(self, input_format, output_format):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.output_directory = Settings.get_instance().output_directory

    def convert(self, model, model_name, trace_numpy_input, logger):
        print(f'convert model from {self.input_format} to {self.output_format}')

    def get_conversion_info(self, model_name, model_size, conversion_time):
        return ConversionInfo(model_name,
                              self.input_format,
                              self.output_format,
                              model_size,
                              conversion_time).get_json_object()
