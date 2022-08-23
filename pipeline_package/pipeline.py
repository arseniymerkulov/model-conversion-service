from torchvision import transforms
from PIL import Image
import numpy as np
import zipfile
import torch
import glob
import os
import shutil


from model_set_package.model_set import ModelSet
from logger_package.json_logger import JsonLogger
from settings_package.settings import Settings


class Pipeline:
    def __init__(self):
        settings = Settings.get_instance()

        self.model_zoo = []
        self.extractor = []
        self.converters = []
        self.verifiers = []
        self.saver = []

        self.input_size = settings.input_size
        self.input_mean = settings.input_mean
        self.input_std = settings.input_std
        self.input_type = settings.input_type
        self.threshold = settings.threshold

        self.trace_numpy_input = self.load_image(glob.glob(f'{settings.images_directory}/*.jpg')[0])
        self.device = torch.device('cpu')

        self.output_directory = settings.output_directory
        self.path_to_log = f'{self.output_directory}/log.json'

        self.logger = JsonLogger(self.path_to_log)

    def load_image(self, path):
        image = Image.open(path)

        preprocess = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.input_mean, std=self.input_std),
        ])

        image = preprocess(image)
        image = image.unsqueeze(0)

        return np.array(image).astype(self.input_type)

    def get_model_path(self, model_filename):
        model_path = os.path.join(self.output_directory, model_filename)
        model_name = os.path.splitext(model_filename)[0]

        return model_path, model_name

    def add_extractor(self, extractor):
        if len(self.extractor) == 0:
            self.extractor.append(extractor)

    def add_converter(self, converter):
        length = len(self.converters)

        if length > 0:
            if self.converters[length - 1].output_format == converter.input_format:
                self.converters.append(converter)
        else:
            self.converters.append(converter)

    def add_verifier(self, verifier):
        self.verifiers.append(verifier)

    def add_saver(self, saver):
        if len(self.saver) == 0:
            self.saver.append(saver)

    def run_model(self, model_filename):
        model_path, model_name = self.get_model_path(model_filename)
        extraction_ret, extraction_err, model, model_type, labels = self.extract_model(model_path, model_name)

        if not extraction_ret:
            return False, f'could not extract model, {extraction_err}', ''

        pipeline_ret, pipeline_err = self.convert_model(model,
                                                        model_name,
                                                        self.extractor[0].input_format,
                                                        model_type,
                                                        labels)

        if not pipeline_ret:
            return False, f'model did not pass pipeline, {pipeline_err}', ''

        output_model_filename = self.save_model()
        return True, '', output_model_filename

    def extract_model(self, model_archive_path, model_name):
        path_to_folder = f'{self.output_directory}/{model_name}'

        if not zipfile.is_zipfile(model_archive_path):
            err = 'file is not an archive'
            print(f'in extract_model: {err}')
            return False, str(err), None, None, None

        with zipfile.ZipFile(model_archive_path, 'r') as model_archive:
            model_archive.extractall(path_to_folder)

        try:
            model, model_type, labels = self.extractor[0].extract(model_name, self.logger)

            # delete model archive and extracted files after loading
            shutil.rmtree(path_to_folder)
            os.remove(model_archive_path)

            return True, '', model, model_type, labels

        except Exception as e:
            # delete model archive and extracted files anyway
            shutil.rmtree(path_to_folder)
            os.remove(model_archive_path)

            print(f'in extract_model: {e}')
            return False, str(e), None, None, None

    def convert_model(self, model, model_name, model_format, model_type, labels):
        model_set = ModelSet(model, model_name, model_format, model_type, labels)

        try:
            for converter in self.converters:
                model_set.convert(converter, self.trace_numpy_input, self.logger)

            for verifier in self.verifiers:
                model_set.verify(verifier, self.trace_numpy_input, self.logger)

            self.model_zoo.append(model_set)
            return True, ''

        except Exception as e:
            print(f'in run_model: {e}')
            return False, str(e)

    def save_model(self):
        self.logger.log()

        log_name = os.path.basename(self.path_to_log)
        log_path = self.path_to_log

        model_set = self.model_zoo[len(self.model_zoo) - 1]
        model = model_set.models[self.saver[0].output_format]
        model_name = model_set.info.model_name

        model_path = self.saver[0].save(model,
                                        model_name,
                                        model_set.labels,
                                        self.get_description())
        model_filename = os.path.basename(model_path)

        archive_name = f'{os.path.splitext(model_filename)[0]}.zip'
        archive_path = f'{self.output_directory}/{archive_name}'

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, model_filename)
            archive.write(log_path, log_name)

        os.remove(model_path)
        os.remove(log_path)

        return archive_name

    def get_description(self):
        model_set = self.model_zoo[len(self.model_zoo) - 1]

        return {
            'name': model_set.info.model_name,
            'description': {
                'modelType': model_set.info.model_type,
                'scoreDivider': 1
            },
            'framework': self.saver[0].output_format,
            'source': model_set.info.initial_format,
            'devices': 'all',
            'version': 1,
            'url': ''
        }

    def clear_output_archive(self):
        archives = glob.glob(f'{self.output_directory}/*.zip')

        for archive in archives:
            os.remove(archive)

