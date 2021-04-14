"""

ml.py module

This module aims at loading all ml models used in the prediction pipeline.

"""




import os

from dataclasses import dataclass

import joblib
import biovec as bv


@dataclass
class LoadModels(object):

    """
        Loads paths to all models into the module.

        Raises:
            FileNotFoundError
    """

    try:

        path_biovec: str = os.path.join(os.getcwd(), 'models/biovec/uniprot2vec.model')
        path_scaler: str = os.path.join(os.getcwd(), 'models/scalers/RobustScaler.model')
        path_umap: str = os.path.join(os.getcwd(), 'models/umap/umap.model')
        path_nn: str = os.path.join(os.getcwd(), 'models/supervised/nn.model')
        path_rf: str = os.path.join(os.getcwd(), 'models/supervised/rf.model')
        path_knn: str = os.path.join(os.path.join(os.getcwd(), 'models/supervised/knn.model'))
        path_xgboost: str = os.path.join(os.getcwd(), 'models/supervised/xgboost.model')
        path_ada: str = os.path.join(os.getcwd(), 'models/supervised/ada_rf.model')
    
    except FileNotFoundError as e:
        raise(e)

    def get_nlp(self):
        """
            Biovec (Protvec) model, has been trained using UniProt DB.

            Returns BioVec (ProtVec) model.
        """
        return bv.models.load_protvec(self.path_biovec)
    
    def get_scaler(self):
        """
            Returns sklearn's RobustScaler.
        """
        return joblib.load(self.path_scaler)
    
    def get_umap(self):
        """
            Returns UMAP model.
        """
        return joblib.load(self.path_umap)
    
    def get_rf(self):
        """
            Returns Random Forest model.
        """
        return joblib.load(self.path_rf)
    
    def get_ada(self):
        """
            Return AdaBoost + Random Forest model.
        """
        return joblib.load(self.path_ada)
    
    def get_knn(self):
        """
            Return k-NN model.
        """
        return joblib.load(self.path_knn)
    
    def get_xg(self):
        """
            Return XGBoost model.
        """
        return joblib.load(self.path_xgboost)
    
    def get_nn(self):
        """
            Return MultiLayer Perceptron model
        """
        return joblib.load(self.path_nn)