# coding=utf8

import os
import json
import torch
import codecs
import argparse
import numpy as np
import six.moves.cPickle as pickle
from rule import define_rule
from utils import load_dataset, epoch_acc
from model import Seq2Tree as TableSeq2Tree


def define_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    arg_parser.add_argument('--model', type=str, help="Path to saved checkpoint")
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='table',
                            help='model name')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--encode_dependency', action='store_true', help='encode dependency')
    arg_parser.add_argument('--encode_entity', action='store_true', help='encode entity')
    arg_parser.add_argument('--dataset', default="/home/v-zezhan/Seq2Tree/data", type=str)
    arg_parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary',
                            default="/home/v-zezhan/Seq2Tree/data/vocab.bin")
    arg_parser.add_argument('--pos_tag', type=str, help='path of the pos tag dictionary', default='')
    arg_parser.add_argument('--table_vocab', type=str, help='path of the serialized table vocabulary',
                            default="/home/v-zezhan/Seq2Tree/data/table_vocab.bin")
    # Model
    arg_parser.add_argument('--decoder_heads', type=int, default=4, help='num heads in Transformer Decoder')
    arg_parser.add_argument('--encoder_heads', type=int, default=4, help='num heads in Transformer Encoder')
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

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--pos_tag_embed_size', default=10, type=int, help='size of pos tag embeddings')
    arg_parser.add_argument('--entity_embed_size', default=5, type=int, help='size of entity tag embeddings')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=32, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.5, type=float, help='word dropout on decoder')
    arg_parser.add_argument('--kl_anneal', default=False, action='store_true')
    arg_parser.add_argument('--alpha', default=0.1, type=float)
    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')
    arg_parser.add_argument('--Root_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--N_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--A_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--Sel_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--Fil_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--Sup_embed_size', default=64, type=int, help='size of attentional vector')
    arg_parser.add_argument('--Ord_embed_size', default=64, type=int, help='size of attentional vector')
    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    return arg_parser


def build_model(args, vocab, grammar, **kwargs):
    model = TableSeq2Tree(args, vocab, grammar, kwargs['table_vocab'])
    return model


def eval_model(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    vocab = pickle.load(open('./vocab.pkl', 'rb'))
    table_vocab = vocab

    if args.pos_tag and len(args.pos_tag) > 0:
        with open(args.pos_tag, 'r', encoding='utf8') as f:
            pos_tags = json.load(f)
    else:
        pos_tags = None

    is_sketch = False
    # if args.model_name == 'sketch':
    #     is_sketch = True
    grammar = define_rule.Grammar(is_sketch=is_sketch)

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=False)

    beam_size = args.beam_size
    print('beam size is ' + str(beam_size))
    batch_size = args.batch_size

    # print(vocab.source.is_unk('citizenship'))
    # quit()
    model = build_model( args, vocab, grammar,
                        table_vocab=table_vocab, pos_tags=pos_tags,
                        is_encode_dependency=args.encode_dependency,
                        is_encode_entity=args.encode_entity)

    model.load_state_dict(torch.load(args.model,
                                     map_location=(lambda storage, loc: storage.cuda(0)) if args.cuda else 'cpu'))
    model.eval()
    if args.cuda:
        model.cuda()
    acc, beam_acc, (right, wrong, beam_wrong), write_data = epoch_acc(model, batch_size, val_sql_data, val_table_data,
                                                                      schemas,
                                                                      beam_size=beam_size,
                                                                      is_sketch=is_sketch,
                                                                      has_pos_tags=pos_tags is not None,
                                                                      is_encode_dependency=args.encode_dependency,
                                                                      is_encode_entity=args.encode_entity)

    print("Acc: %f, Beam Acc: %f" % (acc, beam_acc,))
    save_path, _ = os.path.split(args.model)
    with codecs.open(os.path.join(save_path, 'right.txt'), 'w', encoding='utf-8') as f:
        for a, b, c, d in right:
            f.write(a + '\n')
            f.write(b + '\n')
            f.write(c + '\n')
            f.write(d + '\n')
            f.write('\n')

    with codecs.open(os.path.join(save_path, 'wrong.txt'), 'w', encoding='utf-8') as f:
        for a, b, c, d in wrong:
            f.write(a + '\n')
            f.write(b + '\n')
            f.write(c + '\n')
            f.write(d + '\n')
            f.write('\n')

    with codecs.open(os.path.join(save_path, 'beam_wrong.txt'), 'w', encoding='utf-8') as f:
        for n in beam_wrong:
            f.write('\n'.join(n))
            f.write('\n\n')


if __name__ == "__main__":
    parser = define_arguments()
    args = parser.parse_args()
    print(args)
    eval_model(args)
