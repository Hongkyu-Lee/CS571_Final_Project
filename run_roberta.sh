#!/bin/bash

python main_SC.py --model gcn
python main_SC.py --model gat
python main_SC.py --model sgc
python main_SC.py --model appnp
python main_SC.py --model gcn --train new --test orig
python main_SC.py --model gat --train new --test orig
python main_SC.py --model sgc --train new --test orig
python main_SC.py --model appnp --train new --test orig
