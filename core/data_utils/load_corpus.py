import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


"""
Finalize all data and get them ready so that Dataset at graph_data.py can use them.

1. Get adj_matrix (from build_grap.py)
2. Convert corpus using BERT (NOT HERE)

"""

def load_corpus(basepath):
    