import enum


class ConversionFormat(enum.Enum):
    torch = 'torch'
    onnx = 'onnx'
    keras = 'keras'
    tensorflow = 'tensorflow'
    tensorflow_lite = 'tensorflow_lite'
