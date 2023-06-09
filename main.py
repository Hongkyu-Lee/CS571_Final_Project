import os
import torch
from argparse import ArgumentParser
from distutils.util import strtobool

from core.data_utils import get_data
from core.data_utils import TextGraphData
from core.trainer import Trainer
from core.models import model_selector

import warnings
warnings.filterwarnings("ignore")


parser = ArgumentParser(description="BertGCN on Contextual dataset")
parser.add_argument("--basepath", type=str, default="./data/sentiment", help="path to the dataset")
parser.add_argument("--datasetname", type=str, default="orig", help="Dataset name")
parser.add_argument("--epochs", type=int, default=30, help="epochs")
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--transformer", type=str, default="Bert")
parser.add_argument("--gnn", type=str, default="GCN")
parser.add_argument("--savepath", type=str, default="./save/")
parser.add_argument("--max_length", type=int, default=128, help='the input length for bert')
parser.add_argument("--m", type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument("--eval_int", type=int, default=1, help='Evaluation interval')
parser.add_argument("--use_gpu", default=True, type=lambda x: bool(strtobool(x)))
parser.add_argument("--gpu", type=str, default="cuda:0")
parser.add_argument("--gcn_lr", type=float, default=1e-3)
parser.add_argument("--bert_lr", type=float, default=1e-5)

def main(args):

    # print("1")
    # 1. Define dataset
    x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = get_data(args.basepath, args.datasetname)

    # print("2")
    # 2. Define a model
    if args.transformer.lower() == 'bert':
        hf_transformer_name = 'bert-base-uncased'
    if args.transformer.lower() == 'roberta':
        hf_transformer_name = 'roberta-base'
    if args.transformer.lower() == 'simcse': 
        hf_transformer_name = 'princeton-nlp/unsup-simcse-bert-base-uncased'
    model = model_selector(args.gnn)(pretrained_model=hf_transformer_name,
                                       nb_class=2)

    # print("3")
    # 3. Setup graph data
    txg = TextGraphData(adj, x_tr, y_tr, x_val, y_val, x_ts, y_ts, x_all, y_all, 16)
    
    # print("4")
    #4. Setup training module
    trainer = Trainer(model, txg, vars(args))

    # 4.1. Load corpus  
    with open(os.path.join(args.basepath, args.datasetname,
                           "processed", f"{args.datasetname}_clean_all.txt"), "r") as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
    
    # 4.2. Process corpus using Bert
    print(f'processing corpus using transformers={args.transformer}...')
    _input = model.tokenizer(text, max_length=args.max_length,
                             truncation=True, padding='max_length',
                             return_tensors='pt')
    input_ids, attention_mask = _input.input_ids, _input.attention_mask
    # print(input_ids.shape, attention_mask.shape)
    input_ids = torch.cat([input_ids[:-txg.nb_test],
                           torch.zeros((txg.nb_word, args.max_length), dtype=torch.long),
                           input_ids[-txg.nb_test:]])
    attention_mask = torch.cat([attention_mask[:-txg.nb_test],
                                torch.zeros((txg.nb_word, args.max_length), dtype=torch.long),
                                attention_mask[-txg.nb_test:]])
    # print(input_ids.shape, attention_mask.shape)
    print(f'Number of docs+words = {input_ids.shape[0]}')

    # 4.3 Build DGL graph
    print('buidling DGL heterogeneous graph...')
    txg.build_dgl_graph(input_ids, attention_mask, model.feat_dim)

    # 5. Training  
    print(f'start training model with transformer={args.transformer} and gnn={args.gnn}...')
    trainer.train(model)



if __name__ == "__main__":
    args = parser.parse_args()
    # print("Parsed")
    main(args)