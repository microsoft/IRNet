# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : args.py
# @Software: PyCharm
"""

import random
import argparse
import torch
import numpy as np

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--lr_scheduler', action='store_true', help='use learning rate scheduler')
    arg_parser.add_argument('--lr_scheduler_gammar', default=0.5, type=float, help='decay rate of learning rate scheduler')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=20, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=0.2, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn',
                            help='model name')

    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')

    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', default="glove.42B.300d.txt", type=str)

    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')

    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=128, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=100, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=100, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')


    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')


    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--toy', action='store_true',
                            help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    arg_parser.add_argument('--dataset', default="./data", type=str)

    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='./', type=str,
                            help="Path to save the checkpoint and logs of epoch")

    return arg_parser

def init_config(arg_parser):
    args = arg_parser.parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))
    random.seed(int(args.seed))
    return args
