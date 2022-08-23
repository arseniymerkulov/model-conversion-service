import glob
import zipfile
import json
import os


def clear_directory(directory):
    files = glob.glob(f'{directory}/*.tflite') + \
            glob.glob(f'{directory}/*.json') + \
            glob.glob(f'{directory}/*.csv')

    for file in files:
        os.remove(file)


def log_to_csv(log_record, delimiter=';'):
    result = ''

    for field in log_record:
        result = f'{result}{log_record[field]}{delimiter}'

    return result


if __name__ == '__main__':
    output_dir = 'report/output'

    clear_directory(output_dir)
    clear_directory('report')
    models = glob.glob('output/*.zip')

    for model in models:
        with zipfile.ZipFile(model, 'r') as model_archive:
            model_archive.extractall(output_dir)

        log_path = f'{output_dir}/log.json'

        with open(log_path, 'r') as log:
            log_json = json.load(log)

            for element in log_json:
                save_path = f'report/{element["process_type"]}.csv'

                with open(save_path, 'a') as save_file:
                    save_file.write(log_to_csv(element))
                    save_file.write('\n')

