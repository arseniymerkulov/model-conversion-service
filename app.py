from flask import Flask, request, redirect, send_file
from werkzeug.utils import secure_filename

from session_package.session import Session


session = Session()
app = Flask(__name__)


@app.route('/')
def index():
    return {'message': 'model conversion service'}


@app.route('/api/upload', methods=['POST'])
def upload():
    if session.authorization and not session.check_token(request.headers['Authorization']):
        return {'error': 'unauthorized'}

    session.upload_preprocessing()

    if 'model' not in request.files:
        return {'error': 'no model part'}

    model = request.files['model']

    if model.filename == '':
        return {'error': 'model filename is empty'}

    if model and session.check_file_extension(model.filename):
        filename = secure_filename(model.filename)
        path, _ = session.get_model_path(model.filename)

        model.save(path)

        return redirect(f'/pipeline/{filename}')

    else:
        return {'error': 'invalid model extension'}


@app.route('/pipeline/<model_filename>')
def model_pipeline(model_filename):
    if session.authorization and not session.check_token(request.headers['Authorization']):
        return {'error': 'unauthorized'}

    ret, err, output_model_name = session.upload_postprocessing(model_filename)
    return redirect(f'/api/download/{output_model_name}') if ret else {'error': err}


@app.route('/api/download/<model_filename>')
def model_download(model_filename):
    if session.authorization and not session.check_token(request.headers['Authorization']):
        return {'error': 'unauthorized'}

    model_path, _ = session.get_model_path(model_filename)
    return send_file(model_path)
