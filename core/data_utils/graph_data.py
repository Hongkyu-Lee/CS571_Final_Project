import torch
import torch.nn.functional as F
import torch.utils.data as Data
import scipy.sparse as sp
import numpy as np
import dgl

from core.data_utils.utils import sample_mask, normalize_adj

class TextGraphData:

    def __init__(self, adj, x_tr, y_tr, x_vl, y_vl,
                 x_ts, y_ts, x_all, y_all, batch_size):
        
        # print(adj.shape, x_tr.shape, y_tr.shape, x_vl.shape, y_vl.shape,
        #          x_ts.shape, y_ts.shape, x_all.shape, y_all.shape)
        print(f'Number of train samples = {x_tr.shape[0]}')
        print(f'Number of val samples = {x_vl.shape[0]}')
        print(f'Number of test samples = {x_ts.shape[0]}')
        features = sp.vstack((x_all, x_ts)).tolil()
        # print("Features shape: ", features.shape)
        print(f"Features length = {features.shape[1]}")

        # Load idx
        # we are not shuffling the data so just get them. 

        y_tr = y_tr.toarray()
        y_vl = y_vl.toarray()
        y_ts = y_ts.toarray()
        y_all = y_all.toarray()

        labels = np.vstack((y_all, y_ts))
        # print(labels.shape)
        print(f'Number of classes = {labels.shape[1]}')

        idx_train = range(len(y_tr))
        idx_val = range(len(idx_train), len(idx_train)+len(y_vl))
        idx_test = range(x_all.shape[0], x_all.shape[0]+len(y_ts))

        self.train_mask = sample_mask(idx_train, labels.shape[0])
        self.val_mask = sample_mask(idx_val , labels.shape[0])
        self.test_mask = sample_mask(idx_test, labels.shape[0])
        self.doc_mask = self.train_mask + self.val_mask + self.test_mask

        self.nb_node = features.shape[0]
        self.nb_train = self.train_mask.sum()
        self.nb_val = self.val_mask.sum()
        self.nb_test = self.test_mask.sum()
        self.nb_word = self.nb_node - self.nb_train - self.nb_val - self.nb_test


        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[self.train_mask, :] = labels[self.train_mask, :]
        y_val[self.val_mask, :] = labels[self.val_mask, :]
        y_test[self.test_mask, :] = labels[self.test_mask, :]

        self.y_max = (y_train + y_val + y_test).argmax(axis=1)
        self.y_train_max = y_train.argmax(axis=1)
        self.adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.feature_size = features.shape[0]

        # create index loader
        self.train_idx = Data.TensorDataset(torch.arange(0, self.train_mask.sum(), dtype=torch.long))
        self.val_idx = Data.TensorDataset(torch.arange(self.train_mask.sum(),
                                                  self.train_mask.sum() + self.val_mask.sum(),
                                                  dtype=torch.long))
        self.test_idx = Data.TensorDataset(torch.arange(self.feature_size-self.test_mask.sum(),
                                                   self.feature_size, dtype=torch.long))
        self.doc_idx = Data.ConcatDataset([self.train_idx, self.val_idx, self.test_idx])

        self.idx_loader_train = Data.DataLoader(self.train_idx, batch_size=batch_size, shuffle=True)
        self.idx_loader_val = Data.DataLoader(self.val_idx, batch_size=batch_size)
        self.idx_loader_test = Data.DataLoader(self.test_idx, batch_size=batch_size)
        self.idx_loader = Data.DataLoader(self.doc_idx, batch_size=batch_size, shuffle=True)
        
        self.G = None # Placeholder

    
    def build_dgl_graph(self, input_ids, attention_mask, feat_dim):
        
        # Graph Setting
        # Graph Setting
        # Graph Setting
        adj_norm = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.G = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
        
        self.G.ndata['input_ids'] = input_ids
        self.G.ndata['attention_mask'] = attention_mask
        self.G.ndata['label'] = torch.LongTensor(self.y_max)
        self.G.ndata['train'] = torch.FloatTensor(self.train_mask)
        self.G.ndata['val'] = torch.FloatTensor(self.val_mask)
        self.G.ndata['test'] = torch.FloatTensor(self.test_mask)
        self.G.ndata['label_train'] = torch.LongTensor(self.y_train_max)
        self.G.ndata['cls_feats'] = torch.zeros((self.feature_size, feat_dim))


class SCTextGraphData:

    def __init__(self, adj, x_tr, y_tr, x_vl, y_vl,
                 x_ts, y_ts, x_all, y_all, batch_size):

        print(adj.shape, x_tr.shape, y_tr.shape, x_vl.shape, y_vl.shape,
                 x_ts.shape, y_ts.shape, x_all.shape, y_all.shape)
        features = sp.vstack((x_all, x_ts)).tolil()
        print("Features shape: ", features.shape)

        # Load idx
        # we are not shuffling the data so just get them. 

        y_tr = y_tr.toarray()
        y_vl = y_vl.toarray()
        y_ts = y_ts.toarray()
        y_all = y_all.toarray()

        labels = np.vstack((y_all, y_ts))
        print(labels.shape)

        idx_train = range(len(y_tr))
        idx_val = range(len(idx_train), len(idx_train)+len(y_vl))
        idx_test = range(x_all.shape[0], x_all.shape[0]+len(y_ts))

        
        # Orig and new has same number of samples for both train/valid/test set.

        sub_train_size = int(len(y_tr)/2)
        sub_val_size = int(len(y_vl)/2)
        sub_test_size = int(len(y_ts)/2)

        prev_end = 0
        idx_train_orig = range(sub_train_size)
        prev_end += sub_train_size
        idx_train_new = range(prev_end, prev_end+sub_train_size)
        prev_end += sub_train_size
        idx_val_orig = range(prev_end, prev_end+sub_val_size)
        prev_end += sub_val_size
        idx_val_new = range(prev_end, prev_end+sub_val_size)
        prev_end += sub_val_size
        idx_test_orig = range(prev_end, prev_end+sub_test_size)
        prev_end += sub_test_size
        idx_test_new = range(prev_end, prev_end+sub_test_size)
        

        self.train_mask = sample_mask(idx_train, labels.shape[0])
        self.val_mask = sample_mask(idx_val , labels.shape[0])
        self.test_mask = sample_mask(idx_test, labels.shape[0])
        self.doc_mask = self.train_mask + self.val_mask + self.test_mask

        self.train_orig_mask = sample_mask(idx_train_orig, labels.shape[0])
        self.train_new_mask = sample_mask(idx_train_new, labels.shape[0])
        self.val_orig_mask = sample_mask(idx_val_orig , labels.shape[0])
        self.val_new_mask = sample_mask(idx_val_new , labels.shape[0])
        self.test_orig_mask = sample_mask(idx_test_orig, labels.shape[0])
        self.test_new_mask = sample_mask(idx_test_new, labels.shape[0])
        

        self.nb_node = features.shape[0]
        self.nb_train = self.train_mask.sum()
        self.nb_val = self.val_mask.sum()
        self.nb_test = self.test_mask.sum()
        self.nb_word = self.nb_node - self.nb_train - self.nb_val - self.nb_test


        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[self.train_mask, :] = labels[self.train_mask, :]
        y_val[self.val_mask, :] = labels[self.val_mask, :]
        y_test[self.test_mask, :] = labels[self.test_mask, :]

        self.y_max = (y_train + y_val + y_test).argmax(axis=1)
        self.y_train_max = y_train.argmax(axis=1)
        self.adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.feature_size = features.shape[0]

        # create index loader
        self.train_idx = Data.TensorDataset(torch.arange(0, self.train_mask.sum(), dtype=torch.long))
        self.val_idx = Data.TensorDataset(torch.arange(self.train_mask.sum(),
                                                  self.train_mask.sum() + self.val_mask.sum(),
                                                  dtype=torch.long))
        self.test_idx = Data.TensorDataset(torch.arange(self.feature_size-self.test_mask.sum(),
                                                   self.feature_size, dtype=torch.long))
        self.doc_idx = Data.ConcatDataset([self.train_idx, self.val_idx, self.test_idx])

        self.idx_loader_train = Data.DataLoader(self.train_idx, batch_size=batch_size, shuffle=True)
        self.idx_loader_val = Data.DataLoader(self.val_idx, batch_size=batch_size)
        self.idx_loader_test = Data.DataLoader(self.test_idx, batch_size=batch_size)
        self.idx_loader = Data.DataLoader(self.doc_idx, batch_size=batch_size, shuffle=True)
        

        # create index loader
        prev_end=0
        self.train_orig_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.train_orig_mask.sum()))
        prev_end += self.train_orig_mask.sum()
        self.train_new_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.train_new_mask.sum()))
        prev_end += self.train_new_mask.sum()
        self.val_orig_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.val_orig_mask.sum()))
        prev_end += self.val_orig_mask.sum()
        self.val_new_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.val_new_mask.sum()))
        prev_end += self.val_new_mask.sum()
        self.test_orig_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.test_orig_mask.sum()))
        prev_end += self.test_orig_mask.sum()
        self.test_new_idx = Data.TensorDataset(torch.arange(prev_end, prev_end+self.test_new_mask.sum()))

        self.idx_loader_train_orig = Data.DataLoader(self.train_orig_idx, batch_size=batch_size, shuffle=True)
        self.idx_loader_train_new  = Data.DataLoader(self.train_new_idx,  batch_size=batch_size, shuffle=True)
        self.idx_loader_val_orig   = Data.DataLoader(self.val_orig_idx,   batch_size=batch_size, shuffle=True)
        self.idx_loader_val_new    = Data.DataLoader(self.val_new_idx,    batch_size=batch_size, shuffle=True)
        self.idx_loader_test_orig  = Data.DataLoader(self.test_orig_idx,  batch_size=batch_size, shuffle=True)
        self.idx_loader_test_new   = Data.DataLoader(self.test_new_idx,   batch_size=batch_size, shuffle=True)
        
        self.G = None # Placeholder

    
    def build_dgl_graph(self, input_ids, attention_mask, feat_dim):
        
        # Graph Setting
        # Graph Setting
        # Graph Setting
        adj_norm = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.G = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
        
        self.G.ndata['input_ids'] = input_ids
        self.G.ndata['attention_mask'] = attention_mask
        self.G.ndata['label'] = torch.LongTensor(self.y_max)
        self.G.ndata['train'] = torch.FloatTensor(self.train_mask)
        self.G.ndata['val'] = torch.FloatTensor(self.val_mask)
        self.G.ndata['test'] = torch.FloatTensor(self.test_mask)
        self.G.ndata['label_train'] = torch.LongTensor(self.y_train_max)
        self.G.ndata['cls_feats'] = torch.zeros((self.feature_size, feat_dim))