
import uuid
from typing import Dict

from flask import Flask, request

from api.controller import ReadGB
from api.controller import Analysis


app = Flask(__name__)

@app.route('/random_forest', methods=['POST'])
def rf() -> Dict:
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, probability = analysis.rf_analysis()


    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'bgc_class': bgc_class, 'prob': probability}


@app.route('/adaboost_random_forest', method=['POST'])
def ada():
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, probability = analysis.ada_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'bgc_class': bgc_class, 'prob': probability}

@app.route('/xgboost', method=['POST'])
def xgboost():
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, probability = analysis.xg_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'bgc_class': bgc_class, 'prob': probability}

@app.route('/knn', method=['POST'])
def knn():
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, probability = analysis.knn_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'bgc_class': bgc_class, 'prob': probability}

@app.route('/nn', method=['POST'])
def nn():
    key = uuid.uuid4()
    datum = cluster_vector(request.files)

    analysis = Analysis(datum['bio_vector'])
    bgc_class, probability = analysis.nn_analysis()

    return {'key': key, 'bio_cluster': datum['bio_cluster'], 'bgc_class': bgc_class, 'prob': probability}

def cluster_vector(files) -> Dict:
    readGB = ReadGB(files)
    cluster, vector = readGB.get_data()

    return {'bio_cluster': cluster, 'bio_vector': vector}

if __name__ == '__main__':

    app.run(debug=True)