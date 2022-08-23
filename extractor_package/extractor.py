from model_process_package.model_process import ModelProcess
from settings_package.settings import Settings


class Extractor(ModelProcess):
    def __init__(self, input_format):
        super().__init__()
        self.input_format = input_format
        self.input_directory = Settings.get_instance().output_directory

    def extract(self, model_name, logger):
        print(f'extract model in {self.input_format} format')
