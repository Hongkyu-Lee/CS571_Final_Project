#!/bin/bash

python main_SC.py --model gcn --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model gat --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model sgc --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model appnp --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model gcn --train new --test orig --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model gat --train new --test orig --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model sgc --train new --test orig --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
python main_SC.py --model appnp --train new --test orig --bert bert-base-cased --wandb_name BertGCN-SC-bert-base-GCN
