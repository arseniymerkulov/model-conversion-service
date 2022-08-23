import numpy as np

from model_process_package.model_process import ModelProcess
from verifier_package.verification_info import VerificationInfo
from settings_package.settings import Settings


class Verifier(ModelProcess):
    def __init__(self, input_format, output_format):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.threshold = Settings.get_instance().threshold

    def verify(self, first_model, second_model, model_name, trace_numpy_input, logger):
        print(f'verifying conversion from {self.input_format} to {self.output_format}')

    def get_verification_info(self, model_name, verification_time, max_difference, mismatched_percent):
        return VerificationInfo(model_name,
                                self.input_format,
                                self.output_format,
                                verification_time,
                                self.threshold,
                                max_difference,
                                mismatched_percent).get_json_object()

    def get_mismatched(self, first_output, second_output):
        sizes = [canal.size for canal in first_output]
        diff, miss = self.get_mismatched_list(first_output, second_output)
        return np.max(diff), 100 * np.sum([(miss[i]/100) * sizes[i] for i in range(len(first_output))]) / np.sum(sizes)

    def get_mismatched_list(self, first_output, second_output):
        assert len(first_output) == len(second_output), f'outputs with different length: ' \
                                                        f'{len(first_output)} and ' \
                                                        f'{len(second_output)}'

        diff = [self.get_mismatched_numpy(first_output[i], second_output[i])[0] for i in range(len(first_output))]
        miss = [self.get_mismatched_numpy(first_output[i], second_output[i])[1] for i in range(len(first_output))]

        return diff, miss

    def get_mismatched_numpy(self, first_output, second_output):
        assert first_output.shape == second_output.shape, f'outputs with different shape: ' \
                                                          f'{first_output.shape} and ' \
                                                          f'{second_output.shape}'

        max_difference = np.max(np.abs(first_output - second_output))
        mismatched = first_output.size - np.count_nonzero(np.isclose(first_output, second_output, atol=self.threshold))
        mismatched_percent = 100 * mismatched / first_output.size

        return float(max_difference), float(mismatched_percent)
