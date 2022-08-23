import enum


class ModelType(enum.Enum):
    classification = 'classification'
    detection = 'detection'
    unknown = 'unknown'
