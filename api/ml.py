
import os

from dataclasses import dataclass

import joblib
import biovec as bv


@dataclass
class LoadModels(object):

    path_biovec: str = os.path.join(os.getcwd(), '../models/biovec/uniprot2vec.model')
    path_scaler: str = os.path.join(os.getcwd(), '../models/scalers/RobustScaler.model')
    path_umap: str = os.path.join(os.getcwd(), '../models/umap/umap.model')
    path_nn: str = os.path.join(os.getcwd(), '../models/supervised/nn.model')
    path_rf: str = os.path.join(os.getcwd(), '../models/supervised/rf.model')
    path_knn: str = os.path.join(os.path.join(os.getcwd(), '../models/supervised/knn.model'))
    path_xgboost: str = os.path.join(os.getcwd(), '../models/supervised/xgboost.model')
    path_ada: str = os.path.join(os.getcwd(), '../models/supervised/ada_rf.model')

    def get_nlp(self):
        return bv.models.load_protvec(self.path_biovec)
    
    def get_scaler(self):
        return joblib.load(self.path_scaler)
    
    def get_umap(self):
        return joblib.load(self.path_umap)
    
    def get_rf(self):
        return joblib.load(self.path_rf)
    
    def get_ada(self):
        return joblib.load(self.path_ada)
    
    def get_knn(self):
        return joblib.load(self.path_knn)
    
    def get_xg(self):
        return joblib.load(self.path_xgboost)
    
    def get_nn(self):
        joblib.load(self.path_nn)