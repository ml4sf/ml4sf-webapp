import ast
import os
import subprocess

from flask import Flask, request, render_template, make_response
from werkzeug.exceptions import HTTPException

ALLOWED_EXTENSIONS = {'mol', 'mdl'}

FORM_FILE_INPUT = 'molFile'

ERROR_MSG_NO_FILE = 'No file in request'
ERROR_MSG_EMPTY_FILE = 'No file selected'
ERROR_FILE_TYPE = 'File type not supported'
ERROR_PROCESSING = 'Please try again later'

application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# else mopac won't run as a subprocess (http://openmopac.net/Discussions/Problems%20installing%20MOPAC.html)
os.environ['LD_LIBRARY_PATH'] = '/opt/mopac'
PATH_PYTHON = '/opt/anaconda3/bin/python'
PATH_CALC = 'calc/molecular_properties_calculation.py'

@application.route("/")
def index():
    return make_response(render_template('index.html'))


@application.route('/process', methods=['POST'])
def process():
    # input validations
    if FORM_FILE_INPUT not in request.files:
        return ERROR_MSG_NO_FILE
    file = request.files[FORM_FILE_INPUT]
    if file.filename == '':
        return ERROR_MSG_EMPTY_FILE
    if not allowed_file(file.filename):
        print('!!!!Not allowed')
        return ERROR_FILE_TYPE

    content = file.read().decode('UTF-8')

    proc = subprocess.Popen([PATH_PYTHON, PATH_CALC, content], stdout=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=60)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    res = ast.literal_eval(outs.decode('UTF-8'))

    return render_template('results.html', r=res)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0')
