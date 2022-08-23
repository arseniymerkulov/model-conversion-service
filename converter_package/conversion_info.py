from model_process_package.model_process_info import ModelProcessInfo


class ConversionInfo(ModelProcessInfo):
    def __init__(self, model_name, input_format, output_format, model_size, process_time):
        super().__init__('conversion',
                         model_name,
                         input_format,
                         output_format,
                         model_size,
                         process_time)
