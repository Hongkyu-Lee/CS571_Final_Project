import torch
import transformers
from core.data_utils import SCget_data, SCloadCorpus
from core.models import model_selector
from core.data_utils.build_graph import get_data
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

parser = ArgumentParser(description="BertGCN on Contextual dataset")
parser.add_argument("--basepath", type=str, default="./data/sentiment", help="path to the dataset")
parser.add_argument("--datasetname", type=str, default="orig", help="Dataset name")
parser.add_argument("--epochs", type=int, default=50, help="epochs")
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--model", type=str, default="bert")
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

    #Logging
    run = wandb.init(
        entity="cs571",
        project=args.wandb_name,
        config=args
        )

    x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = SCget_data(args.basepath)
    text = SCloadCorpus(args.basepath)

    y_tr_np = y_tr.toarray()
    y_ts_np = y_ts.toarray()

    y_tr_orig = y_tr_np[:int(len(y_tr_np)/2), :]
    y_tr_new = y_tr_np[int(len(y_tr_np)/2):, :]
    y_ts_orig = y_ts_np[:int(len(y_ts_np)/2), :]
    y_ts_new = y_ts_np[int(len(y_ts_np)/2):, :]

    train_text = text[:x_tr.shape[0]]
    val_text = text[x_tr.shape[0]:x_tr.shape[0] + x_val.shape[0]]
    test_text = text[x_tr.shape[0] + x_val.shape[0]:]

    train_text_orig = train_text[:int(len(y_tr_np)/2)]
    train_text_new  = train_text[int(len(y_tr_np)/2):]
    test_text_orig  = test_text[:int(len(y_ts_np)/2)]
    test_text_new   = test_text[int(len(y_ts_np)/2):]

    print(len(train_text_orig), len(train_text_new), len(test_text_orig), len(test_text_new))

    train_loader = DataLoader((train_text_orig, y_tr_orig), batch_size=32, shuffle=True)
    test_loader = DataLoader((test_text_new, y_ts_new), batch_size=32, shuffle=True)

    model = model_selector(args.model)(nb_class=2, pretrained_model=args.bert, m=args.m)
    model.to(args.gpu)

    optim = torch.optim.Adam([
                    {'params': model.bert_model.parameters(), 'lr': args.bert_lr},
                    {'params': model.classifier.parameters(), 'lr': args.bert_lr}])

    _input = model.tokenizer(text, max_length=args.max_length,
                                    truncation=True, padding='max_length',
                                    return_tensors='pt')
    input_ids, attention_mask = _input.input_ids, _input.attention_mask

    train_orig_iid, train_orig_attm = input_ids[:int(len(y_tr_np)/2)], attention_mask[:int(len(y_tr_np)/2)]
    train_new_iid, train_new_attm = input_ids[int(len(y_tr_np)/2):len(y_tr_np)], attention_mask[int(len(y_tr_np)/2):len(y_tr_np)]
    test_orig_iid, test_orig_attm = input_ids[y_tr.shape[0]+y_val.shape[0]:int(len(y_ts_np)/2)+y_tr.shape[0]+y_val.shape[0]], attention_mask[y_tr.shape[0]+y_val.shape[0]:int(len(y_ts_np)/2)+y_tr.shape[0]+y_val.shape[0]]
    test_new_iid, test_new_attm = input_ids[y_tr.shape[0]+y_val.shape[0]+int(len(y_ts_np)/2):], attention_mask[y_tr.shape[0]+y_val.shape[0]+int(len(y_ts_np)/2):]

    print(len(train_new_iid), len(train_new_attm), len(y_tr_new))
    print(len(test_orig_iid), len(test_orig_attm), len(y_ts_orig))

    if args.train == "orig":
        train_data = TensorDataset(train_orig_iid, train_orig_attm, torch.tensor(y_tr_orig))
        test_data = TensorDataset(test_new_iid, test_new_attm, torch.tensor(y_ts_new))
    elif args.train == "new":
        train_data = TensorDataset(train_new_iid, train_new_attm, torch.tensor(y_tr_new))
        test_data = TensorDataset(test_orig_iid, test_orig_attm, torch.tensor(y_ts_orig))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)


    for e in range(args.epochs):
        train_acc_mean = 0.0
        train_loss_mean = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (ids, attm, y) in enumerate(train_loader):
            model.train()
            ids = ids.to(args.gpu)
            attm = attm.to(args.gpu)
            y = y.to(args.gpu)
            optim.zero_grad()
            pred = model(ids, attm)
            loss = F.nll_loss(pred, torch.argmax(y, axis=1))
            loss.backward()
            optim.step()
            train_loss_mean += loss.item()
            with torch.no_grad():
                model.eval()
                y_true = y.argmax(axis=1).detach().cpu()
                y_pred = pred.argmax(axis=1).detach().cpu()
                train_acc = accuracy_score(y_true, y_pred)
                train_acc_mean += train_acc
            pbar.update()

            
        train_acc_mean /= len(train_loader)
        train_loss_mean /= len(train_loader)
        
        with torch.no_grad():
            for i, (ids, attm, y) in enumerate(test_loader):
                ids = ids.to(args.gpu)
                attm = attm.to(args.gpu)
                pred = model(ids, attm)
                y_true = y.argmax(axis=1).detach().cpu()
                y_pred = pred.argmax(axis=1).detach().cpu()
                test_acc = accuracy_score(y_pred, y_true)
        pbar.close()

        wandb.log({
                "train_loss": train_loss_mean,
                "train_acc" : train_acc_mean,
                "test_acc": test_acc})

    run.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)