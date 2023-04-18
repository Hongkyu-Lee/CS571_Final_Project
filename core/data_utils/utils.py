import os
import re
import csv
import numpy as np
import scipy.sparse as sp

def open_raw_dataset(raw_path, dataset):

    tsv = open(os.path.join(raw_path, dataset), "r")
    lines = csv.reader(tsv, delimiter="\t")
    dataset = list(lines)[1:]

    return dataset


def open_txt_dataset(path):
    lines = list()
    with open(path, "r") as _file:
        for l in _file:
            lines.append(l.strip())
    return lines


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def SCloadCorpus(basepath):
    corpus = list()
    datasetname = "orig"
    dataset = "train"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    print(text)
    corpus += text

    datasetname = "new"
    dataset = "train"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    corpus += text

    datasetname = "orig"
    dataset = "dev"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    corpus += text

    datasetname = "new"
    dataset = "dev"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    corpus += text

    datasetname = "orig"
    dataset = "test"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    corpus += text

    datasetname = "new"
    dataset = "test"
    with open(os.path.join(basepath, datasetname,
                           "processed", f"{datasetname}_clean_{dataset}.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    print(len(text))
    corpus += text

    return corpus


def clean_sentence(string):

    # Remove unnecessary characters
    string = re.sub(r"^\"", "", string)
    string = re.sub(r"\"$", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"<br \/>", "", string)
    string = re.sub(r">", "", string)

    return string.strip().lower()