import requests
import glob
import os


if __name__ == '__main__':
    token = '.'
    url = 'http://127.0.0.1:5000'
    route = '/api/upload'

    headers = {'Authorization': 'Bearer ' + token}

    models = glob.glob('archives/*.zip') + glob.glob('archives/*.mar')

    for model in models:
        req = requests.post(url + route,
                            files={'model': open(model, 'rb')},
                            headers=headers,
                            verify=True)

        with open(f'output/{os.path.splitext(os.path.basename(model))[0]}.zip', 'wb') as file:
            file.write(req.content)
