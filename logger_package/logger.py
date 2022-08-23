class Logger:
    def __init__(self, path):
        self.path = path

    def log_string(self, text):
        with open(self.path, mode='a') as file:
            file.write(text)
            file.write('\n')
