import os
import re
import csv


def open_raw_dataset(raw_path, dataset):

    tsv = open(os.path.join(raw_path, dataset), "r+")
    lines = csv.reader(tsv, delimiter="\t")
    dataset = list(lines)[1:]

    return dataset

def open_txt_dataset(path):
    lines = list()
    with open(path, "w+") as _file:
        for l in _file:
            lines.append(l.strip())
    return lines


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