import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from core.models.GNN.APPNP import APPNP


class BertAPPNP(torch.nn.Module):
    def __init__(self, pretrained_model='roberta-base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=32, 
                 feat_drop=0.5, edge_drop=0.5, alpha=0.5, k=3): 
        super(BertAPPNP, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)
        self.gcn = APPNP(
            num_layers=gcn_layers-1,
            in_feat=self.feat_dim,
            hidden_dim=n_hidden,
            n_classes=nb_class,
            activation=F.elu,
            feat_drop=feat_drop,
            edge_drop=edge_drop,
            alpha=alpha,
            k=k,
        )


    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred

