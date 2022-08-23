import enum


class PipelineType(enum.Enum):
    torch_to_tensorflow_lite = 'torch_to_tensorflow_lite'
    keras_to_tensorflow_lite = 'keras_to_tensorflow_lite'
