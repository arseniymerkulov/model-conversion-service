import importlib
import torch
import glob
import json
import os


from converter_package.conversion_format import ConversionFormat
from model_set_package.model_type import ModelType
from extractor_package.extractor import Extractor


class TorchExtractor(Extractor):
    def __init__(self):
        super().__init__(ConversionFormat.torch.value)

    def extract(self, model_name, logger):
        super().extract(model_name, logger)

        # extract model class
        path_to_folder = f'{self.input_directory}/{model_name}'
        path_to_class = glob.glob(f'{path_to_folder}/*.py')[0]

        class_filename = os.path.splitext(os.path.basename(path_to_class))[0]
        class_module = importlib.import_module(f'{self.input_directory}.{model_name}.{class_filename}')

        try:
            model = class_module.ImageClassifier()
            model_type = ModelType.classification.value
        except AttributeError as e:
            model = class_module.ObjectDetector()
            model_type = ModelType.detection.value

        # extract state dict
        try:
            path_to_weights = glob.glob(f'{path_to_folder}/*.pth')[0]
            model.load_state_dict(torch.load(path_to_weights))
        except Exception as e:
            print(f'in extract_torch_model: could not extract state dict, {e}')

        # extract labels
        labels = []

        try:
            path_to_labels = glob.glob(f'{path_to_folder}/*.json')[0]
            with open(path_to_labels, 'r') as file:
                labels = json.load(file)

        except Exception as e:
            print(f'in extract_torch_model: could not extract labels, {e}')

        return model, model_type, labels
