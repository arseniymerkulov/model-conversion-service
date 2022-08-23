import keras
import glob


from converter_package.conversion_format import ConversionFormat
from model_set_package.model_type import ModelType
from extractor_package.extractor import Extractor


class KerasExtractor(Extractor):
    def __init__(self):
        super().__init__(ConversionFormat.keras.value)

    def extract(self, model_name, logger):
        super().extract(model_name, logger)

        path_to_folder = f'{self.input_directory}/{model_name}'
        model_path = glob.glob(f'{path_to_folder}/*.h5')[0]
        model = keras.models.load_model(model_path)

        return model, ModelType.unknown.value, []
