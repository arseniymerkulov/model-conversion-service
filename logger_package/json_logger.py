import json

from logger_package.logger import Logger


class JsonLogger(Logger):
    def __init__(self, path):
        super().__init__(path)
        self.log_list = []

    def add(self, json_object):
        self.log_list.append(json_object)

    def log(self):
        super().log_string(json.dumps(self.log_list))
        self.log_list.clear()
