import enum


class QuantizationType(enum.Enum):
    none = 'none'
    dynamic = 'dynamic'
    float16 = 'float16'
    full_int = 'full_int'
