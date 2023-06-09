import os
import torch
import torch.nn.functional as F
from torch.utils import data
from core.data_utils.graph_data import TextGraphData
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb



# TBA

class Trainer:

    def __init__(self, model, txg, args):

        self.epochs:int = args["epochs"]
        self.use_gpu:bool = args["use_gpu"]
        self.device = args["gpu"] if self.use_gpu else "cpu"
        self.eval_interval = args["eval_int"]
        self.optim = torch.optim.Adam([
                    {'params': model.bert_model.parameters(), 'lr': args["bert_lr"]},
                    {'params': model.classifier.parameters(), 'lr': args["bert_lr"]},
                    {'params': model.gcn.parameters(), 'lr': args["gcn_lr"]},
                ], lr=1e-3)
        self.scheduler = lr_scheduler.MultiStepLR(self.optim, milestones=[30], gamma=0.1)

        self.train_loss = list()
        self.train_acc = list()
        self.test_loss = list()
        self.valid_metric = list()
        self.test_metric = list()

        self.best_test_acc = 0.0
        self.best_val_acc = 0.0

        self.txg = txg


    def train(self, model):

        
        for epoch in range(self.epochs):
            model.train()
            model.to(self.device)
            self.txg.G = self.txg.G.to(self.device)

            pbar = tqdm(total=len(self.txg.idx_loader_train))
            for i, batch in enumerate(self.txg.idx_loader_train):
                self.optim.zero_grad()
                (idx, ) = [x.to(self.device) for x in batch]
                train_mask = self.txg.G.ndata['train'][idx].type(torch.BoolTensor)
                y_pred = model(self.txg.G, idx)[train_mask]
                y_true = self.txg.G.ndata['label_train'][idx][train_mask]
                loss = F.nll_loss(y_pred, y_true)
                loss.backward()
                self.optim.step()
                self.txg.G.ndata['cls_feats'].detach_()
                train_loss = loss.item()
                with torch.no_grad():
                    if train_mask.sum() > 0:
                        y_true = y_true.detach().cpu()
                        y_pred = y_pred.argmax(axis=1).detach().cpu()
                        train_acc = accuracy_score(y_true, y_pred)
                    else:
                        train_acc = 1
                pbar.set_description(f'Train acc: {train_acc}')
                pbar.update()
            pbar.close()
            
            
            if epoch % self.eval_interval == 0:
                test_acc = self.test(model)
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc" : train_acc,
                    "test_acc": test_acc})

            # Reset Graph
            self.reset_graph(model)


    def validate(self, model, G):

        total_pred_l = list()
        total_true_l = list()
        with torch.no_grad():
            model.eval()
            model = model.to(self.device)
            G = self.txg.G.to(self.device)

            for i, batch in enumerate(self.txg.idx_loader_val):
                (idx, ) = [x.to(self.device) for x in batch]
                y_pred = model(G, idx)
                y_true = self.txg.G.ndata['label'][idx]
                total_pred_l.append(y_pred.clone().cpu())
                total_true_l.append(y_true.clone().cpu())
            
            total_pred = torch.vstack(total_pred_l)
            total_true = torch.vstack(total_true_l)

            acc = accuracy_score(total_true, total_pred)
            f1 = f1_score(total_true, total_pred)
            self.valic_metric.append((acc, f1))
            if acc > self.best_valid_acc:
                self.best_valid_acc = acc

    def test(self, model):

        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            model = model.to(self.device)
            G = self.txg.G.to(self.device)

            for i, batch in enumerate(self.txg.idx_loader_test):
                (idx, ) = [x.to(self.device) for x in batch]
                y_pred = model(G, idx)
                y_true = self.txg.G.ndata['label'][idx]
                correct += torch.sum(torch.argmax(y_pred, dim=1)==y_true)
                total += y_pred.shape[0]
            
            acc = (correct/total).item()
            print("Test acc: ",acc)
            f1 = 0
            self.test_metric.append((acc, f1))
            if acc > self.best_test_acc:
                self.best_test_acc = acc

        return acc


    def update_feature(self, model):
        dataloader = data.DataLoader(
            data.TensorDataset(self.txg.G.ndata['input_ids'][self.txg.doc_mask],
                               self.txg.G.ndata['attention_mask'][self.txg.doc_mask]),
            batch_size=1024
            )
        with torch.no_grad():
            model = model.to(self.device)
            model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
                cls_list.append(output.cpu())
            cls_feat = torch.cat(cls_list, axis=0)
        self.txg.G = self.txg.G.to('cpu')
        self.txg.G.ndata['cls_feats'][self.txg.doc_mask] = cls_feat

    def reset_graph(self, model):
        self.scheduler.step()
        self.update_feature(model)
        torch.cuda.empty_cache()


class SCTrainer:

    def __init__(self, model, txg, args):

        self.epochs:int = args["epochs"]
        self.use_gpu:bool = args["use_gpu"]
        self.device = args["gpu"] if self.use_gpu else "cpu"
        self.eval_interval = args["eval_int"]
        self.model = model
        self.optim = torch.optim.Adam([
                    {'params': model.bert_model.parameters(), 'lr': args["bert_lr"]},
                    {'params': model.classifier.parameters(), 'lr': args["bert_lr"]},
                    {'params': model.gcn.parameters(), 'lr': args["gcn_lr"]},
                ], lr=1e-3)
        self.scheduler = lr_scheduler.MultiStepLR(self.optim, milestones=[30], gamma=0.1)

        self.train_loss = list()
        self.train_acc = list()
        self.test_loss = list()
        self.valid_metric = list()
        self.test_metric = list()

        self.best_test_acc = 0.0
        self.best_val_acc = 0.0

        self.txg = txg

        self.model_save_path = os.path.join(args["savepath"],
                                            args["run_name"],
                                            args["model"]+".pt")
        os.makedirs(os.path.join(args["savepath"], args["run_name"]), exist_ok=True)


        if args["train"] == "orig":
            self.train_loader = self.txg.idx_loader_train_orig
        elif args["train"] == "new":
            self.train_loader = self.txg.idx_loader_train_new
        else:
            raise ValueError(args["train"])

        if args["valid"] == "orig":
            self.valid_loader = self.txg.idx_loader_val_orig
        elif args["valid"] == "new":
            self.valid_loader = self.txg.idx_loader_val_new
        else:
            raise ValueError(args["valid"])
        
        if args["test"] == "orig":
            self.test_loader = self.txg.idx_loader_test_orig
        elif args["test"] == "new":
            self.test_loader = self.txg.idx_loader_test_new
        else:
            raise ValueError(args["test"])
        

    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            self.model.to(self.device)
            self.txg.G = self.txg.G.to(self.device)

            pbar = tqdm(total=len(self.train_loader))
            for i, batch in enumerate(self.train_loader):
                self.optim.zero_grad()
                (idx, ) = [x.to(self.device) for x in batch]
                train_mask = self.txg.G.ndata['train'][idx].type(torch.BoolTensor)
                y_pred = self.model(self.txg.G, idx)[train_mask]
                y_true = self.txg.G.ndata['label_train'][idx][train_mask]
                loss = F.nll_loss(y_pred, y_true)
                loss.backward()
                self.optim.step()
                self.txg.G.ndata['cls_feats'].detach_()
                train_loss = loss.item()
                with torch.no_grad():
                    if train_mask.sum() > 0:
                        y_true = y_true.detach().cpu()
                        y_pred = y_pred.argmax(axis=1).detach().cpu()
                        train_acc = accuracy_score(y_true, y_pred)
                    else:
                        train_acc = 1
                pbar.update()
            pbar.close()
            

            if epoch % self.eval_interval == 0:
                test_acc = self.test()
            
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc" : train_acc,
                    "test_acc": test_acc})

            # Reset Graph
            self.reset_graph()
    
        # save model
        torch.save(self.model.state_dict(), self.model_save_path)


    def validate(self, model, G):

        total_pred_l = list()
        total_true_l = list()
        with torch.no_grad():
            model.eval()
            model = model.to(self.device)
            G = self.txg.G.to(self.device)

            for i, batch in enumerate(self.valid_loader):
                (idx, ) = [x.to(self.device) for x in batch]
                y_pred = model(G, idx)
                y_true = self.txg.G.ndata['label'][idx]
                total_pred_l.append(y_pred.clone().cpu())
                total_true_l.append(y_true.clone().cpu())
            
            total_pred = torch.vstack(total_pred_l)
            total_true = torch.vstack(total_true_l)

            acc = accuracy_score(total_true, total_pred)
            f1 = f1_score(total_true, total_pred)
            self.valic_metric.append((acc, f1))
            if acc > self.best_valid_acc:
                self.best_valid_acc = acc
    

    def test(self):

        correct = 0
        total = 0
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)
            G = self.txg.G.to(self.device)

            for i, batch in enumerate(self.test_loader):
                (idx, ) = [x.to(self.device) for x in batch]
                y_pred = self.model(G, idx)
                y_true = self.txg.G.ndata['label'][idx]
                correct += torch.sum(torch.argmax(y_pred, dim=1)==y_true)
                total += y_pred.shape[0]
            
            acc = (correct/total).item()
            print("Test acc: ",acc)
            f1 = 0
            self.test_metric.append((acc, f1))
            if acc > self.best_test_acc:
                self.best_test_acc = acc
        
        return acc


    def update_feature(self):
        dataloader = data.DataLoader(
            data.TensorDataset(self.txg.G.ndata['input_ids'][self.txg.doc_mask],
                               self.txg.G.ndata['attention_mask'][self.txg.doc_mask]),
            batch_size=1024
            )
        with torch.no_grad():
            self.model = self.model.to(self.device)
            self.model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                output = self.model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
                cls_list.append(output.cpu())
            cls_feat = torch.cat(cls_list, axis=0)
        self.txg.G = self.txg.G.to('cpu')
        self.txg.G.ndata['cls_feats'][self.txg.doc_mask] = cls_feat

    def reset_graph(self):
        self.scheduler.step()
        self.update_feature()
        torch.cuda.empty_cache()


        


     


