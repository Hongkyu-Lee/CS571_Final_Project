from core.models.BertGAT import BertGAT
from core.models.BertGCN import BertGCN
from core.models.BertSGC import BertSGC
from core.models.BertAPPNP import BertAPPNP


def model_selector(model:str):
    if model.lower() == "gcn":
        return BertGCN
    elif model.lower() == "gat":
        return BertGAT
    elif model.lower() == "sgc":
        return BertSGC
    elif model.lower() == "appnp":
        return BertAPPNP
    else:
        raise NotImplementedError

