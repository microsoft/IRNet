#!/bin/bash

devices=$1
save_name=$2

CUDA_VISIBLE_DEVICES=$devices python ./src/train.py --dataset ./data/ \
--vocab ./data/full_data/vocab.bin \
--table_vocab ./data/full_data/table_vocab.bin \
--cuda \
--epoch 60 \
--beam_size 2 \
--model_name table \
--sentence_features \
--column_pointer \
--save $save_name