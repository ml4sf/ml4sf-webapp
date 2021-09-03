from flask import Flask, request, render_template, make_response
import ast

# from test_external.test import func
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mol', 'mdl'}

application = Flask(__name__)
#application = Flask("__name__")
application.config['UPLOAD_FOLDER'] = '/home/rado/ml4sf_webapp/uploads'
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
    content = file.read().decode('UTF-8').replace('\n', '')
    # modif = func(content)
    modif = "hi2  " + content

    ss = "{'timestr': '20210903-090825', 'e_homo': ('0.03441', 'eV'), 'e_lumo': ('0.46412', 'eV'), 'dipole_moment': (0.636, 'D'), 'alpha': (122.8154, 'A.U.'), 'beta': (603.9877, 'A.U.'), 'gamma': (69663.16955, 'A.U.'), 'S1': (5.0297041, 'eV'), 'S2': (5.275003, 'eV'), 'oscillator_strength': (0.072986, ''), 'CI_coef': (0.0, ''), 'T1': (1.7789832999999362, 'eV'), 'T2': (2.923181499999936, 'eV'), 'chemometry_descriptors': {'Si': 150.2665748502994, 'Mv': 0.6211969061673114, 'Mare': 0.9600000000000002, 'Mi': 7.51332874251497, 'nHBAcc': 0, 'nHBDon': 1, 'n6Ring': 1.0, 'n8HeteroRing': 0.0, 'nF9HeteroRing': 1.0, 'nT5HeteroRing': 1.0, 'nT6HeteroRing': 0.0, 'nT7HeteroRing': 0.0, 'nT8HeteroRing': 0.0, 'nT9HeteroRing': 1.0, 'nT10HeteroRing': 0.0, 'nRotB': 0.0, 'RotBFrac': 0.0, 'nRotBt': 1.0, 'RotBtFrac': 0.09090909090909093}}"

    res = ast.literal_eval(ss)
    return render_template('results.html', r = res)


if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0')
