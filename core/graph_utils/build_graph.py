import os
import networkx as nx
import scipy.sparse as sp
from math import log
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import csv
from nltk.corpus import stopwords
import nltk
import re


def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj


def open_raw_dataset(raw_path, dataset):

    tsv = open(os.path.join(raw_path, dataset))
    lines = csv.reader(tsv, delimiter="\t")
    dataset = list(lines)[1:]

    return dataset


def process_sentences(base_path):

    # Create "processed" dir
    os.makedirs(os.path.join(base_path, "processed"), exist_ok=True) 

    _names = ["train.tsv", "test.tsv", "dev.tsv"] # train test valiation

    train_raw = open_raw_dataset(base_path, _names[0])
    test_raw = open_raw_dataset(base_path, _names[1])
    val_raw = open_raw_dataset(base_path, _names[2])

    tvt_split = [len(train_raw), len(val_raw), len(test_raw)]

    total_data = train_raw + test_raw + val_raw

    # split to sentences & labels
    sentneces = list()
    labels = list()

    for entry in total_data:
        sentneces.append(entry[1])
        labels.append(entry[0])

    # save

    with open(os.path.join(base_path, "processed", "sentences.txt"), "w+") as _file:
        for s in sentneces:
            _file.write(f"{s}\n")
    with open(os.path.join(base_path, "processed", "labels.txt"), "w+") as _file:
        for l in labels:
            _file.write(f"{l}\n")
    with open(os.path.join(base_path, "processed", "tvt_idx.txt"), "w+") as _file:
        for s in tvt_split:
            _file.write(f"{s}\n")

    return sentneces, labels, tvt_split


def clean_sentences(base_path, raw_sents=None, raw_labels=None):

    if raw_sents is None:
        # Open processed sentences/labels from the file 

        raw_sents = list()
        raw_labels = list()

        with open(os.path.join(base_path, "processed", "sentences.txt"), "r") as _file:
            for line in _file:
                raw_sents.append(line.strip())
            
        with open(os.path.join(base_path, "processed", "labels.txt"), "r") as _file:
            for line in _file:
                raw_labels.append(line.strip())
    
    # Count the number of words  
    word_count = defaultdict(int)
    
    for sent in raw_sents:
        sent = clean_doc(sent)
        words = sent.split()
        for w in words:
            word_count[w] += 1


    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    sent_clean = list()

    for sent in raw_sents:
        sent = clean_doc(sent)
        words = sent.split()
        words = [word for word in words if word not in stop_words and word_count[word] >= 5]
        sent = " ".join(words).strip()
        if sent != "":
            sent_clean.append(sent)
        else:
            sent_clean.append(" ")

    # save files

    with open(os.path.join(base_path, "processed", "sentences_clean.txt"), "w+") as _file:
        for sent in sent_clean:
            _file.write(f"{sent}\n")
    
    return sent_clean


def clean_doc(string):

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

def load_txt(path):
    _list = list()
    with open(path, "r+") as _file:
        for line in _file:
            _list.append(line.strip())
    
    return _list

def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
            word_freq[word] += 1
    return word_freq

def build_word_doc_edges(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq

def get_adj_matrix(dataset_path):
    """
    Return adj matrix

    """
    
    raw_path = dataset_path
    sent_path = os.path.join(dataset_path, "processed", "sentences.txt")
    sent_cl_path = os.path.join(dataset_path, "processed", "sententes_clean.txt")
    vocab_path = os.path.join(dataset_path, "processed", "vocab.txt")
    label_path = os.path.join(dataset_path, "processed", "label.txt")
    adj_path = os.path.join(dataset_path, "processed", "adj_mat.npz")


    if not os.path.exists(raw_path):
        raise ValueError(f"Dataset path {raw_path} does not exist")

    # 1. Check if sentences have been processed
    if not os.path.exists(sent_path):
        _ = process_sentences(raw_path)

    # 2. Check if sentences have been cleaned
    if not os.path.exists(sent_cl_path):
        sents_cl = clean_sentences(raw_path)

    else:
        # load
        sents_cl = load_txt(sent_cl_path)

    # 3. Check if there is a vocab
    if not os.path.join(vocab_path):
        word_freq = get_vocab(sents_cl)
        vocab = load_txt()
    else:
        word_freq = get_vocab(sents_cl)
        vocab = list(word_freq.keys())
        
        with open(os.path.join(vocab_path), "w+") as _file:
            for v in vocab:
                _file.write(f"{v}\n")
    
    words_in_docs, word_doc_freq = build_word_doc_edges(sents_cl)
    word_id_map = {word: i for i, word in enumerate(vocab)}
    
    # 4. Check if adj_matrix has been generated
    if not os.path.exists(adj_path):

        # 3.1. Vocab 

        adj_mat = build_edges(doc_list=sents_cl,
                        word_id_map=word_id_map,
                        vocab=vocab,
                        word_doc_freq= word_doc_freq
                        )
        sp.save_npz(adj_path, adj_mat)
    else:

        # Return
        adj_mat = sp.load_npz(adj_path)
    return adj_mat

if __name__ == "__main__":
    adj = get_adj_matrix("/home/hongkyu/Documents/projects/2023_CS571/counterfactually-augmented-data/sentiment/orig")
