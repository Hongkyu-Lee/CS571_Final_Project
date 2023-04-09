import os
import random
import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
import sys
from scipy.spatial.distance import cosine
from collections import defaultdict

from core.data_utils.utils import open_raw_dataset, clean_sentence, open_txt_dataset

def process_sentences(basepath, datasetname):
    
    datasets = ("train.tsv", "dev.tsv", "test.tsv")

    train_raw = open_raw_dataset(os.path.join(basepath, datasetname), datasets[0])
    test_raw = open_raw_dataset(os.path.join(basepath, datasetname), datasets[1])
    val_raw = open_raw_dataset(os.path.join(basepath, datasetname), datasets[2])
    
    dataset_files=  (train_raw,test_raw, val_raw)

    x_all_list = list()
    y_all_list = list()
    y_raw = list()


    for name, file in zip(datasets, dataset_files):
        
        raw_sents = list()
        raw_labels = list()

        for (label, sent) in file:
            raw_sents.append(sent.strip())
            raw_labels.append(label)

        word_count = defaultdict(int)

        for sent in raw_sents:
            sent = clean_sentence(sent)
            words = sent.split()
            for w in words:
                word_count[w] += 1

        # Clean sentences
        # Remove irrelevant words and stopwords
        # Use stopwords from NLTK

        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        clean_sents = list()

        for sent in raw_sents:
            sent = clean_sentence(sent)
            words = sent.split()
            words = [word for word in words if word not in stop_words and word_count[word] >= 5]
            sent = " ".join(words).strip()
            if sent != "":
                clean_sents.append(sent)
            else:
                clean_sents.append(" ")

        # Save sentences
        with open(os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_{name[:-4]}.txt"), "w+") as _file:
            for sent in clean_sents:
                _file.write(f"{sent}\n")

        # Save labels:
        with open(os.path.join(basepath, datasetname, "processed", f"{datasetname}_label_{name[:-4]}.txt"), "w+") as _file:
            for l in raw_labels:
                _file.write(f"{l}\n")
        
        if "test" not in name:
            x_all_list.append(clean_sents)          
                  

        # Save labels
        y_all_list.append(raw_labels)

    # Process ALLs
    x_all = list()
    y_all = list()
    for l in x_all_list:
        x_all += l
    for l in y_all_list:
        y_all += l

    with open(os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_all.txt"), "w+") as _file:
        for l in x_all:
            _file.write(f"{l}\n")

    with open(os.path.join(basepath, datasetname, "processed", f"{datasetname}_label_all.txt"), "w+") as _file:
        for l in y_all:
            _file.write(f"{l}\n")


def save_onehot_labels(basepath, filepaths, datasetname):

    # filepaths order: [train, val, test, all]
    datasets = ("train.tsv", "dev.tsv", "test.tsv")

    train_y = open_raw_dataset(os.path.join(basepath, datasetname, "processed"), f"{datasetname}_label_train.txt")
    val_y = open_raw_dataset(os.path.join(basepath, datasetname, "processed"), f"{datasetname}_label_dev.txt")
    test_y = open_raw_dataset(os.path.join(basepath, datasetname, "processed"), f"{datasetname}_label_test.txt")

    y_tr = np.array(train_y)
    y_vl = np.array(val_y)
    y_ts = np.array(test_y)
    y_raw  = [y_tr, y_vl, y_ts]

    # Process labels
    total_label = np.concatenate(y_raw)
    OH = OneHotEncoder()
    OH.fit(total_label.reshape(-1, 1))

    for i, raw_labels in enumerate(y_raw):
        one_hot = OH.transform(raw_labels.reshape(-1, 1))
        print(one_hot.shape)
        f = open(filepaths[i], "wb")
        pickle.dump(one_hot, f)
        f.close()

    y_all  = OH.transform(total_label.reshape(-1, 1))
    f = open(filepaths[-1], "wb")
    pickle.dump(y_all, f)
    f.close()


def count_vocab(x_all:list):
    word_freq = dict()
    word_doc_list = dict()
    word_doc_freq = dict()
    word_id_map = dict()
    word_set = set()

    for line in x_all:
        words = line.split()
        for w in words:
            word_set.add(w)
            
            if w in word_freq:
                word_freq[w] += 1
            else:
                word_freq[w] = 1
    
    vocab= list(word_set)
    vocab_size = len(vocab)

    for idx, line in enumerate(x_all):
        words = line.split()
        for w in words:
            if w in word_doc_list:
                word_doc_list[w].append(idx)
            else:
                word_doc_list[w] = [idx]

    for w, sent_list in word_doc_list.items():
        word_doc_freq[w] = len(sent_list)

    for i, v in enumerate(vocab):
        word_id_map[v] = i

    return vocab, word_doc_freq, word_id_map

def create_and_save_feature(x_data:list, path:str, 
                            word_embeddings_dim:int=300,
                            word_vector_map:dict={}):

    row_x = list()
    col_x = list()
    data_x = list()

    for i, line in enumerate(x_data):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        words = line.split()
        doc_len = len(words)
        for w in words:
            if w in word_vector_map:
                word_vector = word_vector_map[w]
                doc_vec += np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / doc_len)

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(len(x_data), word_embeddings_dim))
    
    # sp.save_npz(path, x)
    f = open(path, 'wb')
    pickle.dump(x, f)
    f.close()

def create_a_graph(x_all_cl, word_id_map, word_doc_freq, vocab, train_size, test_size, path, windowsize:int):
    windows = list()
    for line in x_all_cl:
        words = line.split()
        length = len(words)
        if length <= windowsize:
            windows.append(words)
        else:
            for j in range(length - windowsize+1):
                window = words[j:j+windowsize]
                windows.append(window)
    
    word_window_freq = dict()
    for window in windows:
        appeared = set()
        for i, w in enumerate(window):
            if w in appeared:
                continue
            if w in word_window_freq:
                word_window_freq[w] +=1
            else:
                word_window_freq[w] = 1
            appeared.add(w)

    word_pair_count = dict()
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1           
    
    row = list()
    col = list()
    weight = list()
    vocab_size = len(vocab)

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(x_all_cl)):
        doc_words = x_all_cl[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(x_all_cl)):
        doc_words = x_all_cl[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(x_all_cl) /
                    word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    
    # sp.save_npz(path, adj)
    f = open(path, "wb")
    pickle.dump(adj, f)
    f.close()

def process_data(basepath:str, filepaths:list, datasetname:str, windowsize:int=20):
    
    # directory check

    if not os.path.exists(os.path.join(basepath, datasetname, "processed")):
        os.makedirs(os.path.join(basepath, datasetname, "processed"))

    # Clean sentneces
    process_sentences(basepath, datasetname)

    # Need to save x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all and adj_path

    # Load clean_train / clean_validation / clean_test set
    

    x_tr_cl_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_train.txt")
    x_vl_cl_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_dev.txt")
    x_ts_cl_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_test.txt")
    x_all_cl_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}_clean_all.txt")
    vocab_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}_vocab.txt")

    x_path = filepaths[0]
    tx_path = filepaths[2]
    vx_path = filepaths[4]
    allx_path = filepaths[6]
    adj_path = filepaths[8]
    y_path = filepaths[1]
    ty_path = filepaths[3]
    vy_path = filepaths[5]
    ally_path = filepaths[7]

    x_tr_cl = open_txt_dataset(x_tr_cl_path)
    x_vl_cl = open_txt_dataset(x_vl_cl_path)
    x_ts_cl = open_txt_dataset(x_ts_cl_path)
    x_all_cl = open_txt_dataset(x_all_cl_path)

    # We will skip shuffling the dataset. We can shuffle it at batch-processing step.

    save_onehot_labels(basepath, (y_path, vy_path, ty_path, ally_path), datasetname)

    # Get Vocab
    vocab, word_doc_freq, word_id_map = count_vocab(x_all_cl)
    vocab_size = len(vocab)
    
    # save vocab
    with open(vocab_path, "w+") as _file:
        for v in vocab:
            _file.write(f"{v}\n")

    create_and_save_feature(x_data=x_tr_cl, path=x_path)
    create_and_save_feature(x_data=x_ts_cl, path=tx_path)
    create_and_save_feature(x_data=x_vl_cl, path=vx_path)
    create_and_save_feature(x_data=x_all_cl, path=allx_path)
    
    ### Create a Graph
    create_a_graph(x_all_cl=x_all_cl,
                   word_id_map=word_id_map,
                   word_doc_freq=word_doc_freq,
                   vocab=vocab,
                   train_size=len(x_tr_cl)+len(x_vl_cl),
                   test_size=len(x_ts_cl),
                   path=adj_path,
                   windowsize=windowsize)


def load_data(x_tr_path, y_tr_path, x_ts_path, y_ts_path,
              x_val_path, y_val_path, x_all_path, y_all_path, adj_path):
    _x_tr = open(x_tr_path, "rb")
    _y_tr = open(y_tr_path, "rb")
    _x_ts = open(x_ts_path, "rb")
    _y_ts = open(y_ts_path, "rb")
    _x_val = open(x_val_path, "rb")
    _y_val = open(y_val_path, "rb")
    _x_all = open(x_all_path, "rb")
    _y_all = open(y_all_path, "rb")
    _adj = open(adj_path, "rb")

    x_tr = pickle.load(_x_tr, encoding='latin1')
    y_tr = pickle.load(_y_tr, encoding='latin1')
    x_ts = pickle.load(_x_ts, encoding='latin1')
    y_ts = pickle.load(_y_ts, encoding='latin1')
    x_val = pickle.load(_x_val, encoding='latin1')
    y_val = pickle.load(_y_val, encoding='latin1')
    x_all = pickle.load(_x_all, encoding='latin1')
    y_all = pickle.load(_y_all, encoding='latin1')
    adj = pickle.load(_adj, encoding='latin1')

    _x_tr.close()
    _y_tr.close()
    _x_ts.close()
    _y_ts.close()
    _x_val.close()
    _y_val.close()
    _x_all.close()
    _y_all.close()
    _adj.close()

    return x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj


def get_data(basepath:str, datasetname:str, windowsize:int=20):

    x_tr_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.x_tr.npz")
    y_tr_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.y_tr.npz")
    x_ts_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.x_ts.npz")
    y_ts_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.y_ts.npz")
    x_val_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.x_vl.npz")
    y_val_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.y_vl.npz")
    x_all_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.x_all.npz")
    y_all_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.y_all.npz")
    adj_path = os.path.join(basepath, datasetname, "processed", f"{datasetname}.adj.npz")
    


    if os.path.exists(x_tr_path) and \
       os.path.exists(y_tr_path) and \
       os.path.exists(x_ts_path) and \
       os.path.exists(y_ts_path) and \
       os.path.exists(x_val_path) and \
       os.path.exists(y_val_path) and \
       os.path.exists(x_all_path) and \
       os.path.exists(y_all_path) and \
       os.path.exists(adj_path):

        x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = load_data(x_tr_path,
        y_tr_path, x_ts_path, y_ts_path, x_val_path, y_val_path, x_all_path, y_all_path, adj_path)

    else:

        filepaths = [
            x_tr_path, y_tr_path, x_ts_path, y_ts_path, x_val_path, y_val_path,
            x_all_path, y_all_path, adj_path
        ]

        process_data(basepath,filepaths, datasetname, windowsize)
        x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = load_data(x_tr_path,
        y_tr_path, x_ts_path, y_ts_path, x_val_path, y_val_path, x_all_path, y_all_path, adj_path)

    return x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj
