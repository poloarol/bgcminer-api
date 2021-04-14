
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

    file: IO[str]
    cluster: List[Protein] = field(default_factory=list)
    vector: NDArray = field(default_factory=np.array())
    load_models = LoadModels()

    def __post_init__(self):

        prot_id: str
        locus: str
        product: str
        gene: str
        translation: str
        description: str
        seq: str

        records = SeqIO.read(self.file, 'genbank')

        biovector = self.load_models.get_nlp()

        for i, record in enumerate(records):


            if record.type == 'CDS':

                if 'translation' in record.qualifiers:
                    translation =  record.qualifiers['translation'][0]
                if 'locus_tag' in record.qualifiers:
                    locus = record.qualifiers['locus_tag'][0]
                if 'protein_id' in record.qualifiers:
                    prot_id = record.qualifiers['protein_id'][0]
                if 'product' in record.qualifiers:
                    product = record.qualifiers['product'][0]
                if 'description' in record.qualifiers:
                    description = record.qualifiers['description'][0]
                if 'gene' in record.qualifiers:
                    gene = record.qualifiers['gene'][0]

                strand: int = int(record.strand)
                start, stop = record.location.start.position, record.location.end.position
                seq: str = record.seq[start:stop]

                p = Protein(gene, prot_id, locus, product, seq, translation, [start, stop], strand, description)

                self.cluster.append(p)

                v = biovector.to_vecs(translation)

                if self.vector.size == 0:
                    self.vector = v
                else:
                    self.vector = np.add(v, self.vector)

    
    def get_data(self):
        return self.cluster, self.vector


@dataclass
def Analysis(object):

    vector: NDArray
    load_models = LoadModels()

    def __post_init__(self):
        self.scaler = self.load_models.get_scaler()
        self.umap = self.load_models.get_umap()

        self.vector = self.vector.flatten()
        self.vector = self.scaler.transform()


    def rf_analysis(self):

        try:
            rf = self.load_models.get_rf()
            bgc_class, probability = rf.predict(self.vector), rf.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def ada_analysis(self):

        try:
            ada = self.load_models.get_ada()
            bgc_class, probability = ada.predict(self.vector), ada.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    

    def xg_analysis(self):

        try:
            xgboost = self.load_models.get_xg()
            bgc_class, probability = xgboost.predict(self.vector), xgboost.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def knn_analysis(self):
        try:
            knn = self.load_models.get_knn()
            bgc_class, probability = knn.predict(self.vector), knn.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e
    
    def nn_analysis(self):
        try:
            nn = self.load_models.get_nn()
            bgc_class, probability = nn.predict(self.vector), nn.predict_proba(self.vector)

            return bgc_class, probability
        except Exception as e:
            return e


