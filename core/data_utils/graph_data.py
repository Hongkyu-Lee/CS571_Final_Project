from core.data_utils.build_graph import build_edges

import torch
import torch.nn.functional as F
import torch.utils.data as Data


class TextGraphData(object):

    def __init__(self, adj, feature, y_train, y_val, y_test,
                       train_mask, test_mask, val_mask,
                       train_size, test_size):
    

        self.train_idx = Data.TensorDataset()
        