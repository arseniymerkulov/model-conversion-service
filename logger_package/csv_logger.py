from logger_package.logger import Logger


class CsvLogger(Logger):
    def __init__(self, path, delimiter=';'):
        super().__init__(path)
        self.delimiter = delimiter

    def log(self, csv_list):
        string = ''

        for element in csv_list:
            string = f'{string}{element}{self.delimiter}'

        super().log_string(string)
