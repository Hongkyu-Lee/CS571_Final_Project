from core.models.BertGAT import BertGAT
from core.models.BertGCN import BertGCN

def model_selector(model:str):
    if model.lower() == "bertgcn":
        return BertGCN
    elif model.lower() == "bertgat":
        return BertGAT
    else:
        raise NotImplementedError

