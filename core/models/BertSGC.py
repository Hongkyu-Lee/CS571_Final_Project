"""
Simplifying Graph Convolutional Networks implementation in DGL
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
"""


import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from dgl.nn.pytorch.conv import SGConv
import dgl


class BertSGC(torch.nn.Module):
    def __init__(self, pretrained_model='roberta-base', nb_class=20, m=0.7, k=2):
        super(BertSGC, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)
        self.gcn = SGConv(
            in_feats=self.feat_dim,
            out_feats=nb_class,
            k=k,
            cached=False,
            bias=True,
            norm=None,
            allow_zero_in_degree=False,
        )


    def forward(self, g, idx):
         # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g, g.ndata['cls_feats'])[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred
