import torch
import torch.nn.functional as F
import torch.utils.data as Data
import scipy.sparse as sp
import numpy as np
import dgl

from core.data_utils.utils import sample_mask, normalize_adj

class TextGraphData(object):

    def __init__(self, adj, x_tr, y_tr, x_vl, y_vl,
                 x_ts, y_ts, x_all, y_all, batch_size):

        features = sp.vstack((x_all, x_ts)).tolil()
        labels = np.vstack((x_all, y_tr))

        # Load idx
        # we are not shuffling the data so just get them.  

        idx_train = range(len(y_tr))
        idx_val = range(len(idx_train), len(idx_train)+len(y_vl))
        idx_test = range(x_all.shape[0], x_all.shape[0]+len(y_ts))

        self.train_mask = sample_mask(idx_train, labels.shape[0])
        self.val_mask = sample_mask(idx_val , labels.shape[0])
        self.test_mask = sample_mask(idx_test, labels.shape[0])

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
                                                  self.train_mask.sum() + self.val_maks.sum(),
                                                  dtype=torch.long))
        self.test_idx = Data.TensorDataset(torch.arange(self.feature_size-self.test_mask.sum(),
                                                   self.feature_size, dtype=torch.long))
        self.doc_idx = Data.ConcatDataset([self.train_idx, self.val_idx, self.test_idx])

        self.idx_loader_train = Data.DataLoader(self.train_idx, batch_size=batch_size, shuffle=True)
        self.idx_loader_val = Data.DataLoader(self.val_idx, batch_size=batch_size)
        self.idx_loader_test = Data.DataLoader(self.test_idx, batch_size=batch_size)
        self.idx_loader = Data.DataLoader(self.doc_idx, batch_size=batch_size, shuffle=True)
        
        self.g = None # Placeholder

    
    def build_dgl_graph(self, input_ids, attention_mask, feat_dim):
        
        # Graph Setting
        # Graph Setting
        # Graph Setting
        adj_norm = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
        
        self.g.ndata['input_ids'] = input_ids
        self.g.ndata['attention_mask'] = attention_mask
        self.g.ndata['label'] = torch.LongTensor(self.y)
        self.g.ndata['train'] = torch.FloatTensor(self.train_mask)
        self.g.ndata['val'] = torch.FloatTensor(self.val_mask)
        self.g.ndata['test'] = torch.FloatTensor(self.test_mask)
        self.g.ndata['label_train'] = torch.LongTensor(self.y_train_max)
        self.g.ndata['cls_feats'] = torch.zeros((self.feature_size, feat_dim))
