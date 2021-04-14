
import uuid
import json
from typing import Dict

from flask import Flask, request

from api.controller import ReadGB
from api.controller import Analysis


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'.gb', '.gbk'}

@app.route('/', methods=['GET'])
def index():
    return {'message': 'Hello World'}

@app.route('/random_forest', methods=['POST'])
def rf() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, prob = analysis.rf_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'predicted_class': int(bgc_class[0]), 'pks': prob[0][0], 'nrps': prob[0][1], 'terpene': prob[0][2]}


@app.route('/adaboost_random_forest', methods=['POST'])
def ada() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, prob = analysis.ada_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'predicted_class': int(bgc_class[0]), 'pks': prob[0][0], 'nrps': prob[0][1], 'terpene': prob[0][2]}

@app.route('/xgboost', methods=['POST'])
def xgboost() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, prob = analysis.xg_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'predicted_class': int(bgc_class[0]), 'pks': prob[0][0], 'nrps': prob[0][1], 'terpene': prob[0][2]}

@app.route('/knn', methods=['POST'])
def knn() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, prob = analysis.knn_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'predicted_class': int(bgc_class[0]), 'pks': prob[0][0], 'nrps': prob[0][1], 'terpene': prob[0][2]}

@app.route('/nn', methods=['POST'])
def nn() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, prob = analysis.nn_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'predicted_class': int(bgc_class[0]), 'pks': prob[0][0], 'nrps': prob[0][1], 'terpene': prob[0][2]}


def allowed_extensions(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cluster_vector(files) -> Dict:

    """ Load GenBank file """

    if 'files' not in files:
        return request.url
    
    file = files['files']

    if file.filename == '':
        return request.url
    
    if file and allowed_extensions(file.filename):
        readGB = ReadGB(files['files'])
        cluster, vector = readGB.get_data()

        return {'bio_cluster': cluster, 'bio_vector': vector}

if __name__ == '__main__':

    app.run(debug=True)