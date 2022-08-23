from model_process_package.model_process_info import ModelProcessInfo


class VerificationInfo(ModelProcessInfo):
    def __init__(self,
                 model_name, input_format, output_format, process_time,
                 threshold, max_difference, mismatched_percent):
        super().__init__('verification',
                         model_name,
                         input_format,
                         output_format,
                         None,
                         process_time)

        self.threshold = threshold
        self.max_difference = max_difference
        self.mismatched_percent = mismatched_percent
