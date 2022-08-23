from model_process_package.model_process import ModelProcess
from settings_package.settings import Settings


class Saver(ModelProcess):
    def __init__(self, output_format):
        super().__init__()
        self.output_format = output_format
        self.output_directory = Settings.get_instance().output_directory

    def save(self, model, model_name, labels, description):
        print(f'save model in {self.output_format} format')
