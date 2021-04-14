
import io

from os import scandir
from typing import IO, NamedTuple, List
from collections import namedtuple

from dataclasses import dataclass, field

import numpy as np
from nptyping import NDArray

from Bio import SeqIO

from .ml import LoadModels

Protein: NamedTuple = namedtuple('protein', 'gene, protein_id, locus_tag, product, dna, translation, location, strand, description')

@dataclass
class ReadGB(object):

    """
        Reads a GenBank file and extract information pertaining to each gene.
            - Name, Product, Protein ID, Locus Tag
            - DNA Sequence, Protein Sequence
            - Start, Stop, Coding strand (-1, 1)

        Parameters
        ----------

        file : IO[str]
            GenBank file to analyze
        cluster: List[Protein]
            List of all proteins in the GenBank file
        vector: NP Array
            Vectorial representation of cluster
        load_models
            Reference to LoadModels Class
        
    """

    file: IO[str]
    cluster: List[Protein] = field(default_factory=list)
    vector = np.array([])
    load_models = LoadModels()

    def __post_init__(self):

        prot_id: str
        locus: str
        product: str
        gene: str
        translation: str
        description: str
        seq: str

        tmp_file = self.file.read() # self.file is a `BytesIO` object

        # convert to a 'unicode' string object
        tmp_file = tmp_file.decode() # Or use the encoding you expect


        biovector = self.load_models.get_nlp()

        #convert to io.StringIO, which I think SeqIO will understand well.
        records = SeqIO.parse(io.StringIO(tmp_file), 'genbank')
        for record in records:
            for feature in record.features:

                if feature.type == 'CDS':

                    translation =  feature.qualifiers['translation'][0] if 'translation' in feature.qualifiers else None
                    locus = feature.qualifiers['locus_tag'][0] if 'locus_tag' in feature.qualifiers else None
                    prot_id = feature.qualifiers['protein_id'][0] if 'protein_id' in feature.qualifiers else None
                    product = feature.qualifiers['product'][0] if 'product' in feature.qualifiers else None
                    description = feature.qualifiers['description'][0] if 'description' in feature.qualifiers else None
                    gene = feature.qualifiers['gene'][0] if 'gene' in feature.qualifiers else None

                    strand: int = int(feature.strand)
                    start, stop = feature.location.start.position, feature.location.end.position
                    seq: str = record.seq[start:stop]

                    p = Protein(gene, prot_id, locus, product, str(seq), translation, [start, stop], strand, description)

                    self.cluster.append(p)

                    v = np.array(biovector.to_vecs(translation))

                    if self.vector.size == 0:
                        self.vector = v
                    else:
                        self.vector = np.add(v, self.vector)

    
    def get_data(self):
        """
            Returns all protein information in cluster and vectorial representation of cluster
        """
        return self.cluster, self.vector


@dataclass
class Analysis(object):

    """
        Runs the ML predictions.
        All preprocessing is done in the __post_init__ method
            converting to 1D array
            Scaling (RobustScaler)
            Dimensional Reduction (UMAP)

        Parameter
        ---------

        vector: numpy array representation of the cluster
    """

    vector: NDArray
    load_models = LoadModels()

    def __post_init__(self):
        scaler = self.load_models.get_scaler()
        umap = self.load_models.get_umap()

        self.vector = self.vector.flatten()
        # self.vector = self.vector.reshape(-1, 1)
        self.vector = scaler.transform([self.vector])
        self.vector = umap.transform(self.vector)


    def rf_analysis(self):

        """
            Predicts the most probable cluster using a Random Forest Classifier

            Returns
            -------

            bgc_class : int
                The class in which the cluster belongs
            probabilities: List
                Probability of belonging to each class 
        """

        try:
            rf = self.load_models.get_rf()
            bgc_class, probability = rf.predict(self.vector), rf.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def ada_analysis(self):

        """
            Predicts the most probable cluster using an AdaBoost + Random Forest Classifier

            Returns
            -------

            bgc_class : int
                The class in which the cluster belongs
            probabilities: List
                Probability of belonging to each class 
        """

        try:
            ada = self.load_models.get_ada()
            bgc_class, probability = ada.predict(self.vector), ada.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    

    def xg_analysis(self):

        """
            Predicts the most probable cluster using an XGBoost Classifier

            Returns
            -------

            bgc_class : int
                The class in which the cluster belongs
            probabilities: List
                Probability of belonging to each class 
        """

        try:
            xgboost = self.load_models.get_xg()
            bgc_class, probability = xgboost.predict(self.vector), xgboost.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def knn_analysis(self):

        """
            Predicts the most probable cluster using a k-NN classifier

            Returns
            -------

            bgc_class : int
                The class in which the cluster belongs
            probabilities: List
                Probability of belonging to each class 
        """


        try:
            knn = self.load_models.get_knn()
            bgc_class, probability = knn.predict(self.vector), knn.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def nn_analysis(self):

        """
            Predicts the most probable cluster using a Multilayer Perceptron

            Returns
            -------

            bgc_class : int
                The class in which the cluster belongs
            probabilities: List
                Probability of belonging to each class 
        """


        try:
            nn = self.load_models.get_nn()
            bgc_class, probability = nn.predict(self.vector), nn.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e


