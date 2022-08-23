import time


class ModelProcess:
    def __init__(self):
        self.timers = {}

    def start_timer(self):
        self.timers['start'] = time.time()

    def check_timer(self):
        if 'start' in self.timers:
            return "%.2f" % (time.time() - self.timers['start'])
