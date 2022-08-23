from tflite_support import metadata
import json
import os


from saver_package.saver import Saver
from converter_package.conversion_format import ConversionFormat


class TensorflowLiteSaver(Saver):
    def __init__(self):
        super().__init__(ConversionFormat.tensorflow_lite.value)

    def save(self, model, model_name, labels, description):
        super().save(model, model_name, labels, description)
        path = f'{self.output_directory}/{model_name}.tflite'

        with open(path, 'wb') as file:
            file.write(model)

        self.add_metadata(path, labels, description)

        return path

    def add_metadata(self, model_path, labels, description):
        labels_path = f'{self.output_directory}/labels.txt'
        description_path = f'{self.output_directory}/description.json'

        with open(labels_path, 'w') as file:
            file.write(json.dumps(labels))

        with open(description_path, 'w') as file:
            file.write(json.dumps(description))

        writer = metadata.MetadataPopulator.with_model_file(model_path)
        writer.load_associated_files([labels_path, description_path])
        writer.populate()

        os.remove(labels_path)
        os.remove(description_path)
