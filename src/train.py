# coding=utf-8

from __future__ import print_function
import traceback
import argparse
import six.moves.cPickle as pickle
import numpy as np
import time
import os
import sys
import json
import copy
import torch
from tqdm import tqdm
from collections import OrderedDict
from model import Seq2Tree as TableSeq2Tree
from rule import define_rule
import nn_utils
from utils import GloveHelper, load_dataset, epoch_train, epoch_acc
import random
import torch.optim as optim


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--encode_dependency', action='store_true', help='encode dependency')
    arg_parser.add_argument('--encode_entity', action='store_true', help='encode entity')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'wikisql', 'prolog'], default='python')
    arg_parser.add_argument('--mode', choices=['train', 'self_train', 'train_decoder', 'train_semi', 'log_semi', 'test',
                                               'sample'], default='train', help='run mode')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn',
                            help='model name')
    arg_parser.add_argument('--curriculum', action='store_true', help='curriculum_learning')
    arg_parser.add_argument('--curriculum_step', default=10, type=int, help='curriculum_step')

    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')

    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', type=str)
    arg_parser.add_argument('--glove_300_embed_path', type=str)

    arg_parser.add_argument('--lr_scheduler', action='store_true', help='use learning rate scheduler')
    arg_parser.add_argument('--lr_scheduler_gammar', default=0.5, type=float,
                            help='decay rate of learning rate scheduler')

    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--unsup_batch_size', default=10, type=int)
    arg_parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='sample size')
    arg_parser.add_argument('--pos_tag_embed_size', default=10, type=int, help='size of pos tag embeddings')
    arg_parser.add_argument('--entity_embed_size', default=5, type=int, help='size of entity tag embeddings')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--parent_feeding', action='store_true', help='enable parent feeding')
    arg_parser.add_argument('--encode_sketch', action='store_true', help='encode sketch')
    arg_parser.add_argument('--sketch_history', action='store_true', help='use sketch history')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=20, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=0.2, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--stanford_tokenized', action='store_true', help='use stanford tokenization')

    arg_parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert_model')
    arg_parser.add_argument('--encoder_dim', default=768, type=int, help='size of encoder_dim')

    arg_parser.add_argument('--action_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=32, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')
    arg_parser.add_argument('--kl_anneal', default=False, action='store_true')
    arg_parser.add_argument('--alpha', default=0.1, type=float)


    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true')

    # wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    arg_parser.add_argument('--answer_prune', dest='answer_prune', action='store_true')
    arg_parser.add_argument('--no_answer_prune', dest='answer_prune', action='store_false')
    arg_parser.set_defaults(answer_prune=True)

    # parent information switch and input feeding
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true')
    arg_parser.add_argument('--no_input_feed', default=False, action='store_true')
    arg_parser.add_argument('--no_copy', default=False, action='store_true')

    arg_parser.add_argument('--asdl_file', type=str)
    arg_parser.add_argument('--pos_tag', type=str, help='path of the pos tag dictionary', default='')
    arg_parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary',
                            default="/home/v-zezhan/Seq2Tree/data/vocab.bin")
    arg_parser.add_argument('--table_vocab', type=str, help='path of the serialized table vocabulary',
                            default='/home/v-zezhan/Seq2Tree/data/table_vocab.bin')
    arg_parser.add_argument('--train_src', type=str, help='path to the training source file')
    arg_parser.add_argument('--unlabeled_file', type=str, help='path to the training source file')
    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    arg_parser.add_argument('--test_file', type=str, help='path to the test target file')
    arg_parser.add_argument('--prior_lm_path', type=str, help='path to the prior LM')

    # self-training
    arg_parser.add_argument('--load_decode_results', default=None, type=str)

    # semi-supervised learning arguments
    arg_parser.add_argument('--load_decoder', default=None, type=str)
    arg_parser.add_argument('--load_src_lm', default=None, type=str)

    arg_parser.add_argument('--baseline', choices=['mlp', 'src_lm', 'src_lm_and_linear'], default='mlp')
    arg_parser.add_argument('--prior', choices=['lstm', 'uniform'])
    arg_parser.add_argument('--load_prior', type=str, default=None)
    arg_parser.add_argument('--clip_learning_signal', type=float, default=None)
    arg_parser.add_argument('--begin_semisup_after_dev_acc', type=float, default=0.,
                            help='begin semi-supervised learning after'
                                 'we have reached certain dev performance')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')

    arg_parser.add_argument('--valid_metric', default='sp_acc', choices=['nlg_bleu', 'sp_acc'],
                            help='metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int)
    arg_parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')

    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--toy', action='store_true',
                            help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='save decoding results to file')
    arg_parser.add_argument('--patience', default=5, type=int, help='training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int)
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='if specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int)
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False)
    arg_parser.add_argument('--verbose', action='store_true', default=False)
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='only evaluate the top prediction in validation')

    arg_parser.add_argument('--train_opt', default="reinforce", type=str, choices=['reinforce', 'st_gumbel'])
    arg_parser.add_argument('--dataset', default="/home/v-zezhan/Seq2Tree/data", type=str)

    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='', type=str,
                            help="Path to save the checkpoint and logs of epoch")
    arg_parser.add_argument('--decoder_heads', type=int, default=4, help='num heads in Transformer Decoder')
    arg_parser.add_argument('--encoder_heads', type=int, default=4, help='num heads in Transformer Encoder')

    return arg_parser


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))
    random.seed(int(args.seed))

    return args

def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def init_log_checkpoint_path():
    save_path = args.save
    dir_name = save_path + str(int(time.time()))
    save_path = os.path.join(os.path.curdir, 'saved_model', dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path


def build_model(model_name, args, vocab, grammar, **kwargs):
    if model_name == 'transformer':
        model = TransformerSeq2Tree(args, vocab, grammar)
    elif model_name == 'table':
        model = TableSeq2Tree(args, vocab, grammar, kwargs['table_vocab'])
    else:
        model = None
    return model


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))


def partition_data_based_on_hardness(sql_data):
    hardness = dict()
    for s in sql_data:
        h = s['hardness']
        if h not in hardness:
            hardness[h] = list()
        hardness[h].append(s)
    hardness = OrderedDict(sorted(hardness.items(), key=lambda t: t[0]))
    partitioned_data = list()
    prev_data = None
    for h, d in hardness.items():
        if prev_data is None:
            prev_data = copy.deepcopy(d)
        else:
            prev_data = prev_data + d
        partitioned_data.append(prev_data)
    print([len(x) for x in partitioned_data])
    return partitioned_data


def train(args):
    """
    :param args:
    :return:
    """
    vocab = pickle.load(open('./vocab.pkl', 'rb'))

    if args.pos_tag and len(args.pos_tag) > 0:
        with open(args.pos_tag, 'r', encoding='utf8') as f:
            pos_tags = json.load(f)
    else:
        pos_tags = None

    is_sketch = False
    if args.model_name == 'sketch':
        is_sketch = True

    grammar = define_rule.Grammar(is_sketch=is_sketch)

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=args.toy)


    model = build_model(args.model_name, args, vocab, grammar,
                        table_vocab=vocab, pos_tags=pos_tags,
                        is_encode_dependency=args.encode_dependency,
                        is_encode_entity=args.encode_entity)
    model.train()
    if args.cuda:
        model.cuda()


    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in ['bertModel'])], 'lr': args.lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['bertModel'])], 'lr': args.lr * 0.05}
    ]    # optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr)

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)

    # model initial
    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    # begin train
    num_epoch = args.epoch

    # model.src_embed.weight.requires_grad = False
    beam_size = args.beam_size
    batch_size = args.batch_size
    model_save_path = init_log_checkpoint_path()
    save_args(args, os.path.join(model_save_path, 'config.json'))
    best_dev_acc = .0
    is_curriculum_learning = args.curriculum
    curriculum_step = args.curriculum_step
    try:
        if is_curriculum_learning:
            partitioned_data = partition_data_based_on_hardness(sql_data)
            train_sql_data = partitioned_data[0]
        else:
            partitioned_data = [sql_data]
            train_sql_data = sql_data
        with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
            num = 0
            idx = 0
            for epoch in tqdm(range(num_epoch)):
                if is_curriculum_learning and (epoch - num) == curriculum_step:
                    print('Change')
                    num = epoch
                    if idx < len(partitioned_data) - 1:
                        idx += 1
                        train_sql_data = partitioned_data[idx]

                epoch_begin = time.time()
                loss = epoch_train(model, optimizer, batch_size, train_sql_data, table_data, schemas, args,
                                   is_sketch=is_sketch,
                                   has_pos_tags=pos_tags is not None,
                                   is_encode_dependency=args.encode_dependency,
                                   is_encode_entity=args.encode_entity, epoch=epoch)
                epoch_end = time.time()
                acc, _, (right, wrong, _), write_data = epoch_acc(model, 1, val_sql_data, val_table_data,
                                                                  schemas,
                                                                  beam_size=1,
                                                                  is_sketch=is_sketch,
                                                                  has_pos_tags=pos_tags is not None,
                                                                  is_encode_dependency=args.encode_dependency,
                                                                  is_encode_entity=args.encode_entity)


                if acc > best_dev_acc:
                    save_checkpoint(model, os.path.join(model_save_path, 'best_model.model'))
                    best_dev_acc = acc

                train_acc = 0.0
                log_str = 'Epoch: %d, Loss: %f, Acc: %f, Train Acc: %f, time: %f\n' % (
                    epoch + 1, loss, acc, train_acc, epoch_end - epoch_begin)
                tqdm.write(log_str)
                epoch_fd.write(log_str)
    except Exception as e:
        # Save model
        save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
        print(e)
        tb = traceback.format_exc()
        print(tb)
    else:
        save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
        acc, beam_acc, (right, wrong, _), write_data = epoch_acc(model, batch_size, val_sql_data, val_table_data, schemas,
                                                              beam_size=beam_size,
                                                              is_sketch=is_sketch,
                                                              has_pos_tags=pos_tags is not None,
                                                              is_encode_dependency=args.encode_dependency,
                                                              is_encode_entity=args.encode_entity)
        print("Acc: %f, Beam Acc: %f" % (acc, beam_acc,))


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args)
    train(args)
