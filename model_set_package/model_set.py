from model_set_package.model_set_info import ModelSetInfo
from converter_package.converter import Converter
from verifier_package.verifier import Verifier


class ModelSet:
    def __init__(self, model, model_name, model_format, model_type, labels):
        self.info = ModelSetInfo(model_name, model_type, model_format)
        self.labels = labels
        self.models = {model_format: model}

    def convert(self, converter, trace_numpy_input, logger):
        assert isinstance(converter, Converter), 'converter is not an instance of Converter class'
        assert converter.input_format in self.models, 'converter input format not in model set'

        output = converter.convert(self.models[converter.input_format],
                                   self.info.model_name,
                                   trace_numpy_input,
                                   logger)

        self.models[converter.output_format] = output

    def verify(self, verifier, trace_numpy_input, logger):
        assert isinstance(verifier, Verifier), 'verifier is not an instance of Verifier class'
        assert verifier.input_format in self.models and \
               verifier.output_format in self.models, 'verifier input formats not in model set'

        verifier.verify(self.models[verifier.input_format],
                        self.models[verifier.output_format],
                        self.info.model_name,
                        trace_numpy_input,
                        logger)
