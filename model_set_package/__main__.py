from converter_package.torch_onnx_converter import TorchOnnx
from model_set_package.model_set import ModelSet


def main():
    converter = TorchOnnx()

    model_set = ModelSet(None, 'cpu')
    model_set.convert(converter)


if __name__ == '__main__':
    main()
