class ModelProcessInfo:
    def __init__(self, process_type, model_name, input_format, output_format, model_size, process_time):
        self.process_type = process_type
        self.model_name = model_name
        self.input_format = input_format
        self.output_format = output_format
        self.model_size = model_size
        self.process_time = process_time

    def get_json_object(self):
        return self.__dict__
