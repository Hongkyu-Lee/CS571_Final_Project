import os
import torch
import wandb
from argparse import ArgumentParser
from core.data_utils import SCget_data, SCloadCorpus
from core.data_utils import SCTextGraphData
from core.trainer import SCTrainer, SCBertTrainer
from core.models import model_selector


parser = ArgumentParser(description="BertGCN on Contextual dataset")
parser.add_argument("--basepath", type=str, default="./data/sentiment", help="path to the dataset")
parser.add_argument("--datasetname", type=str, default="orig", help="Dataset name")
parser.add_argument("--epochs", type=int, default=50, help="epochs")
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--model", type=str, default="gcn")
parser.add_argument("--savepath", type=str, default="./save/")
parser.add_argument("--max_length", type=int, default=128, help='the input length for bert')
parser.add_argument("--m", type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument("--eval_int", type=int, default=1, help='Evaluation interval')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--gpu", type=str, default="cuda:1")
parser.add_argument("--gcn_lr", type=float, default=1e-3)
parser.add_argument("--bert_lr", type=float, default=1e-5)
parser.add_argument("--train", type=str, default="orig")
parser.add_argument("--valid", type=str, default="orig")
parser.add_argument("--test", type=str, default="new")
parser.add_argument("--wandb_name", type=str, default="BertGCN-SC-Training")
parser.add_argument("--bert", type=str, default="roberta-base")
parser.add_argument("--logdir", type=str, default="./log")

def main(args):

    # Logging
    run = wandb.init(
        entity="cs571",
        project=args.wandb_name,
        config=args
        )

    print("1")
    # 1. Define dataset
    x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = SCget_data(args.basepath)

    print("2")
    # 2. Define a model
    model = model_selector(args.model)(nb_class=2, pretrained_model=args.bert, m=args.m)

    if args.model.lower() == "bert" or args.model.lower() == 'roberta':
        TRAINER = SCBertTrainer
    else:
        TRAINER = SCTrainer

    print("3")
    # 3. Setup graph data
    txg = SCTextGraphData(adj, x_tr, y_tr, x_val, y_val, x_ts, y_ts, x_all, y_all, 16)
    
    print("4")
    #4. Setup training module
    arg_dict = vars(args)
    arg_dict["run_name"] = run.name
    trainer = TRAINER(model, txg, arg_dict)

    # 4.1. Load corpus
    text = SCloadCorpus(args.basepath)
    # print(len(text))  

    # 4.2. Process corpus using Bert
    _input = model.tokenizer(text, max_length=args.max_length,
                             truncation=True, padding='max_length',
                             return_tensors='pt')
    input_ids, attention_mask = _input.input_ids, _input.attention_mask
    print(input_ids.shape, attention_mask.shape)
    input_ids = torch.cat([input_ids[:-txg.nb_test],
                           torch.zeros((txg.nb_word, args.max_length), dtype=torch.long),
                           input_ids[-txg.nb_test:]])
    attention_mask = torch.cat([attention_mask[:-txg.nb_test],
                                torch.zeros((txg.nb_word, args.max_length), dtype=torch.long),
                                attention_mask[-txg.nb_test:]])
    print(input_ids.shape, attention_mask.shape)

    # 4.3 Build DGL graph
    txg.build_dgl_graph(input_ids, attention_mask, model.feat_dim)

    # 5. Training  
    trainer.train()

    run.finish()



if __name__ == "__main__":
    args = parser.parse_args()
    print("Parsed")
    main(args)