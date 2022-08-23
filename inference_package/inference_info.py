from model_process_package.model_process_info import ModelProcessInfo


class InferenceInfo(ModelProcessInfo):
    def __init__(self, model_name, input_format, process_time):
        super().__init__('inference',
                         model_name,
                         input_format,
                         input_format,
                         None,
                         process_time)
