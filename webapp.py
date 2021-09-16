from flask import Flask, request, render_template, make_response
import os, subprocess, sys, ast

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mol', 'mdl'}

application = Flask(__name__)
#application = Flask("__name__")
#application.config['UPLOAD_FOLDER'] = '/home/rado/ml4sf_webapp/uploads'
application.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route("/")
def form():
    r = make_response(render_template('index.html'))
    r.headers.set('Content-Language', "en-US")
    return r


@application.route('/process', methods=['POST'])
def my_form_post():
    if 'molFile' not in request.files:
        return 'No file in request'

    file = request.files['molFile']
    if file.filename == '':
        return 'No file selected'
    if not allowed_file(file.filename):
        return 'File type not supported'

    # if file and allowed_file(file.filename):
    # filename = secure_filename(file.filename)
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # return redirect(url_for('download_file', name=filename))
    content = file.read().decode('UTF-8')

    os.environ['LD_LIBRARY_PATH'] = '/opt/mopac'

    proc = subprocess.Popen(['/opt/anaconda3/bin/python', 'calc/molecular_properties_calculation.py', content], stdout=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=60)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    res = ast.literal_eval(outs.decode('UTF-8'))
    return render_template('results.html', r = res)


if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0')
