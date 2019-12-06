# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import json
import time

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer

from src.dataset import Example
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()


def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    return ret

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1

def process(sql, table):

    process_dict = {}

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))

    col_set_type = np.zeros((len(col_set_iter), 4))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names

    return process_dict

def is_valid(rule_label, col_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
            except:
                flag = True
                print(sql['question'])
    return flag is False


def to_batch_seq(sql_data, table_data, idxes, st, ed,
                 is_train=True):
    """

    :return:
    """
    examples = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql['db_id']]

        process_dict = process(sql, table)

        for c_id, col_ in enumerate(process_dict['col_set_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_set_type'][c_id][0] += 1

        schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                       process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], sql)

        col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
        table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

        process_dict['col_set_iter'][0] = ['count', 'number', 'many']

        rule_label = None
        if 'rule_label' in sql:
            try:
                rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
            except:
                continue
            if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
                continue

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_set_iter']),
            vis_seq=(sql['question'], process_dict['col_set_iter'], sql['query']),
            tab_cols=process_dict['col_set_iter'],
            sql=sql['query'],
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['col_set_type'],
            tgt_actions=rule_label
        )
        example.sql_json = copy.deepcopy(sql)
        examples.append(example)

    if is_train:
        examples.sort(key=lambda e: -len(e.src_sent))
        return examples
    else:
        return examples

def epoch_train(model, optimizer, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2):
    model.train()
    # shuffe
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed)
        optimizer.zero_grad()

        score = model.forward(examples)
        loss_sketch = -score[0]
        loss_lf = -score[1]

        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)

        if epoch > loss_epoch_threshold:
            loss = loss_lf + sketch_loss_coefficient * loss_sketch
        else:
            loss = loss_lf + loss_sketch

        loss.backward()
        if args.clip_grad > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        st = ed
    return cum_loss / len(sql_data)

def epoch_acc(model, batch_size, sql_data, table_data, beam_size=3):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0

    json_datas = []
    sketch_correct, rule_label_correct, total = 0, 0, 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed,
                                                        is_train=False)
        for example in examples:
            results_all = model.parse(example, beam_size=beam_size)
            results = results_all[0]
            list_preds = []
            try:

                pred = " ".join([str(x) for x in results[0].actions])
                for x in results:
                    list_preds.append(" ".join(str(x.actions)))
            except Exception as e:
                # print('Epoch Acc: ', e)
                # print(results)
                # print(results_all)
                pred = ""

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] =  " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == simple_json['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == simple_json['model_result']:
                rule_label_correct += 1
            total += 1

            json_datas.append(simple_json)
        st = ed
    return json_datas, float(sketch_correct)/float(total), float(rule_label_correct)/float(total)

def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1
    print(best_correct / len(preds))
    return best_correct / len(preds)


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    table_data_new = {table['db_id']: table for table in table_data}

    if use_small:
        return sql_data[:80], table_data_new
    else:
        return sql_data, table_data_new


def load_dataset(dataset_dir, use_small=False):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    DEV_PATH = os.path.join(dataset_dir, "dev.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)

    return train_sql_data, train_table_data, val_sql_data, val_table_data


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def init_log_checkpoint_path(args):
    save_path = args.save
    dir_name = save_path + str(int(time.time()))
    save_path = os.path.join(os.path.curdir, 'saved_model', dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path
