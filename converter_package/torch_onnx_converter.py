import onnxsim
import torch
import onnx
import os

from converter_package.converter import Converter
from converter_package.conversion_format import ConversionFormat


class TorchOnnxConverter(Converter):
    def __init__(self):
        super().__init__(ConversionFormat.torch.value, ConversionFormat.onnx.value)

    def convert(self, model, model_name, trace_numpy_input, logger):
        super().convert(model, model_name, trace_numpy_input, logger)
        path = f'{self.output_directory}/{model_name}.onnx'

        self.start_timer()
        model.eval()

        torch.onnx.export(model, torch.from_numpy(trace_numpy_input), path,
                          do_constant_folding=True,
                          export_params=True,
                          verbose=False,
                          opset_version=11,
                          # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          input_names=['input']
                          )

        model = onnx.load(path)

        model, check = onnxsim.simplify(model)
        assert check, "simplified Onnx model could not be validated"

        # save and load model to check model file size
        onnx.save(model, path)
        model_size = os.stat(path).st_size
        model = onnx.load(path)
        os.remove(path)

        conversion_time = self.check_timer()
        logger.add(self.get_conversion_info(model_name, model_size, conversion_time))

        return model
