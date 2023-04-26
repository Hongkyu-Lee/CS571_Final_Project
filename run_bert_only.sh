#!/bin/bash

python bert_SC.py --model bert --bert bert-base-cased --wandb_name BertGCN-SC-bert-only --m 0.0 --train orig
python bert_SC.py --model bert --bert roberta-base --wandb_name BertGCN-SC-bert-only --m 0.0 --train orig
python bert_SC.py --model bert --bert bert-base-cased --wandb_name BertGCN-SC-bert-only --m 0.0 --train new
python bert_SC.py --model bert --bert roberta-base --wandb_name BertGCN-SC-bert-only --m 0.0 --train new