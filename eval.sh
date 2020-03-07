#!/bin/bash

devices=$1
model_path=$2

CUDA_VISIBLE_DEVICES=$devices python ./src/eval.py --dataset ./data --vocab ./data/vocab.bin --cuda \
--beam_size 5 \
--model $model_path \
--batch_size 1 \
--sentence_features \
