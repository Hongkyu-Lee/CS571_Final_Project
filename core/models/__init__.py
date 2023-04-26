from core.models.BertGAT import BertGAT
from core.models.BertGCN import BertGCN
from core.models.BertSGC import BertSGC
from core.models.BertAPPNP import BertAPPNP
from core.models.BERT import BertClassifier


def model_selector(model:str):
    if model.lower() == "gcn":
        return BertGCN
    elif model.lower() == "gat":
        return BertGAT
    elif model.lower() == "sgc":
        return BertSGC
    elif model.lower() == "appnp":
        return BertAPPNP
    if model.lower() == "bert" or model.lower() == "roberta":
        return BertClassifier
    else:
        raise NotImplementedError

