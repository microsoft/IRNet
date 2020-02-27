# coding=utf-8
import argparse

import os
import lf
import torch
import numpy as np
import six.moves.cPickle as pickle
from nltk.stem import WordNetLemmatizer
from dataset import Example
from components.vocab import VocabEntry
from rule.define_rule import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
import collections
import json
import re as regex

import random as normal_random

AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']

wordnet_lemmatizer = WordNetLemmatizer()
from bert_utils import generate_inputs
import copy
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)


class GloveHelper(object):
    def __init__(self, glove_file, embedding_size=100):
        self.glove_file = glove_file
        embeds = np.zeros((5000, embedding_size), dtype='float32')
        for i, (word, embed) in enumerate(self.embeddings):
            if i == 5000: break
            embeds[i] = embed

        self.mean = np.mean(embeds)
        self.std = np.std(embeds)

    @property
    def embeddings(self):
        with open(self.glove_file, 'r', encoding='utf8') as f:
            for line in f:
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]])
                yield word, embed

    def emulate_embeddings(self, shape):
        samples = np.random.normal(self.mean, self.std, size=shape)

        return samples

    def get_weights(self, vocab, embedding_size):
        word_ids = set(range(len(vocab.source)))
        numpy_embed = np.zeros(shape=(len(vocab.source), embedding_size))
        for word, embed in self.embeddings:
            if word in vocab.source:
                word_id = vocab.source[word]
                word_ids.remove(word_id)
                numpy_embed[word_id] = embed
        word_ids = list(word_ids)
        numpy_embed[word_ids] = self.emulate_embeddings(shape=(len(word_ids), embedding_size))
        return numpy_embed

    def load_to(self, embed_layer, vocab, trainable=True):

        word_ids = set(range(embed_layer.num_embeddings))
        numpy_embed = np.zeros(shape=(embed_layer.weight.shape[0], embed_layer.weight.shape[1]))
        for word, embed in self.embeddings:
            if word in vocab:
                word_id = vocab[word]
                word_ids.remove(word_id)
                numpy_embed[word_id] = embed

        word_ids = list(word_ids)
        numpy_embed[word_ids] = self.emulate_embeddings(shape=(len(word_ids), embed_layer.embedding_dim))
        embed_layer.weight.data.copy_(torch.from_numpy(numpy_embed))
        embed_layer.weight.requires_grad = trainable

    @property
    def words(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                yield tokens[0]

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    sql_data_new, table_data_new = process(sql_data, table_data)  # comment out if not on full dataset

    schemas = {}
    for tab in table_data:
        schemas[tab['db_id']] = tab

    if use_small:
        return sql_data_new[:80], table_data_new, schemas
    else:
        return sql_data_new, table_data_new, schemas


def load_dataset(dataset_dir, use_small=False):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    DEV_PATH = os.path.join(dataset_dir, "dev.json")
    TEST_PATH = os.path.join(dataset_dir, "dev.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data, schemas_all = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data, schemas = load_data_new(DEV_PATH, table_data, use_small=use_small)
    test_sql_data, test_table_data, schemas = load_data_new(TEST_PATH, table_data, use_small=use_small)

    TRAIN_DB = '../alt/data/train.db'
    DEV_DB = '../alt/data/dev.db'
    TEST_DB = '../alt/data/test.db'

    return train_sql_data, train_table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, schemas_all, TRAIN_DB, DEV_DB, TEST_DB


def process(sql_data, table_data):
    output_tab = {}
    tables = {}
    tabel_name = set()
    # remove_list = ['?', '.', ',', "''", '``', '(', ')', "'"]
    remove_list = list()

    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['schema_len'] = []
        length = {}
        for col_tup in temp['col_map']:
            length[col_tup[0]] = length.get(col_tup[0], 0) + 1
        for l_id in range(len(length)):
            temp['schema_len'].append(length[l_id-1])
        temp['foreign_keys'] = table['foreign_keys']
        temp['primary_keys'] = table['primary_keys']
        temp['table_names'] = table['table_names']
        temp['column_types'] = table['column_types']
        db_name = table['db_id']
        tabel_name.add(db_name)
        output_tab[db_name] = temp
        tables[db_name] = table
    output_sql = []
    for i in range(len(sql_data)):
        sql = sql_data[i]
        sql_temp = {}

        # add query metadata
        for key, value in sql.items():
            sql_temp[key] = value
        sql_temp['question'] = sql['question']

        sql_temp['question_tok'] = [wordnet_lemmatizer.lemmatize(x).lower() for x in sql['question_toks'] if x not in remove_list]
        sql_temp['rule_label'] = sql['rule_label']
        sql_temp['col_set'] = sql['col_set']
        sql_temp['query'] = sql['query']
        # dre_file.write(sql['query'] + '\n')
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']
        table = tables[sql['db_id']]
        sql_temp['col_org'] = table['column_names_original']
        sql_temp['table_org'] = table['table_names_original']
        sql_temp['table_names'] = table['table_names']
        sql_temp['fk_info'] = table['foreign_keys']
        tab_cols = [col[1] for col in table['column_names']]
        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
        sql_temp['col_iter'] = col_iter
        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1]) #GOLD for sel and agg

        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])
                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond) #GOLD for COND [[col, op],[]]

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']] #assume only one groupby
        having_cond = []
        if len(sql['sql']['having']) > 0:
            gt_having = sql['sql']['having'][0] # currently only do first having condition
            having_cond.append([gt_having[2][1][0]]) # aggregator
            having_cond.append([gt_having[2][1][1]]) # column
            having_cond.append([gt_having[1]]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        sql_temp['group'].append(having_cond) #GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # process order by / limit
        order_aggs = []
        order_cols = []
        sql_temp['order'] = []
        order_par = 4
        gt_order = sql['sql']['orderby']
        limit = sql['sql']['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]] # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        sql_temp['order'] = [order_aggs, order_cols, order_par] #GOLD for ORDER [[[agg], [col], [dat]], []]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect'] is not None:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        if 'stanford_tokenized' in sql:
            sql_temp['stanford_tokenized'] = sql['stanford_tokenized']
        if 'stanford_pos' in sql:
            sql_temp['stanford_pos'] = sql['stanford_pos']
        if 'stanford_dependencies' in sql:
            sql_temp['stanford_dependencies'] = sql['stanford_dependencies']
        if 'hardness' in sql:
            sql_temp['hardness'] = sql['hardness']
        if 'question_labels' in sql:
            sql_temp['question_labels'] = sql['question_labels']

        output_sql.append(sql_temp)
    return output_sql, output_tab


def to_batch_seq(sql_data, table_data, idxes, st, ed, schemas,
                 is_train=True,
                 is_sketch=False,
                 has_pos_tags=False,
                 is_encode_dependency=False,
                 is_encode_entity=False,
                 move_place=0):
    """
    :param sql_data:
    :param table_data:
    :param idxes:
    :param st:
    :param ed:
    :param schemas:
    :return:
    """
    examples = []
    col_org_seq = []
    schema_seq = []

    # file = codecs.open('./type.txt', 'w', encoding='utf-8')
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql['table_id']]
        origin_sql = sql['question_toks']
        table_names = table['table_names']
        col_org_seq.append(sql['col_org'])
        schema_seq.append(schemas[sql['table_id']])
        tab_cols = [col[1] for col in table['col_map']]
        tab_ids = [col[0] for col in table['col_map']]

        q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]



        col_set = sql['col_set']

        col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]

        question_arg = copy.deepcopy(sql['question_arg'])
        col_set_type = np.zeros((len(col_set_iter), 4))


        for c_id, col_ in enumerate(col_set_iter):

            for q_id, ori in enumerate(q_iter_small):
                if ori in col_:
                    col_set_type[c_id][0] += 1


        question_arg_type = sql['question_arg_type']
        one_hot_type = np.zeros((len(question_arg_type), 6))

        another_result = []
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
                        another_result.append(sql['col_set'].index(col_probase))
                else:
                    for col_probase in t_q:
                        if col_probase == 'asd':
                            continue
                        col_set_type[sql['col_set'].index(col_probase)][3] += 1

        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]



        table_dict = {}
        for c_id, c_v in enumerate(col_set):
            for cor_id, cor_val in enumerate(tab_cols):
                if c_v == cor_val:
                    table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

        col_table_dict = { }
        for key_item, value_item in table_dict.items():
            for value in value_item:
                col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
        col_table_dict[0] = [x for x in range(len(table_dict) - 1)]

        col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]

        table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table_names]


        q_iter_small = [wordnet_lemmatizer.lemmatize(x) for x in origin_sql]


        table_set_type = np.zeros((len(table_names), 1))

        for c_id, col_ in enumerate(table_names):
            if " ".join(col_) in q_iter_small or " ".join(col_) in " ".join(q_iter_small):
                table_set_type[c_id][0] = 5
                continue
            for q_id, ori in enumerate(q_iter_small):
                if ori in col_:
                    table_set_type[c_id][0] += 1
                    # col_hot_type[c_id][6] = q_id + 1

        try:
            rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
        except:
            continue

        flag = False
        for r_id, rule in enumerate(rule_label):
            if type(rule) == C:
                try:
                    assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
                except:
                    flag = True
                    # print(sql['question'])
        if flag:
            continue

        table_col_name = get_table_colNames(tab_ids, col_iter)

        if has_pos_tags:
            pos_tags = sql['stanford_pos']
        else:
            pos_tags = None

        if is_encode_dependency:
            dependency_graph_adjacency_matrix = normalize_adjacency_matrix(sql['stanford_dependencies'])
        else:
            dependency_graph_adjacency_matrix = None

        entities = None
        if is_encode_entity:
            entities = sql['question_labels']

        pattern = regex.compile('C\(.*?\)')
        result_pattern = set(pattern.findall(sql['rule_label']))
        ground_col_labels = []
        for c in result_pattern:
            index = int(c[2:-1])
            ground_col_labels.append(index)

        if is_train is True:

            # print(col_labels)
            # col_labels = sql['col_pred'][:len(ground_col_labels)]
            col_labels = []
            for label_val in ground_col_labels:
                if label_val not in col_labels:
                    col_labels.append(label_val)
            sample_col = normal_random.sample([x for x in range(len(col_set_iter))], min(7, len(col_set_iter)))
            c_count = 0
            # if move_place < 15:
            #     thred = 3
            # elif move_place < 22:
            #     thred = 6
            # else:
            #     thred = 10
            thred = 3

            for c in sample_col:
                if c not in ground_col_labels and c_count < thred:
                    c_count += 1
                    col_labels.append(c)
            # print(col_labels)

        else:
            col_labels = sql['col_pred'][:len(ground_col_labels) + 1]

        col_set_iter_remove = [col_set_iter[x] for x in sorted(col_labels)]
        remove_dict = {}
        for k_i, k in enumerate(sorted(col_labels)):
            remove_dict[k] = k_i

        # now should change the rule label
        # if is_train is False:
        #     print('asdasd')
        for label in rule_label:
            if type(label) == C:
                if label.id_c not in remove_dict:
                    remove_dict[label.id_c] = len(remove_dict)
                label.id_c = remove_dict[label.id_c]

        col_set_iter = col_set_iter_remove

        remove_dict_reverse = {}
        for k, v in remove_dict.items():
            remove_dict_reverse[v] = k


        enable_table = []
        col_table_dict_remove = {}
        for k, v in col_table_dict.items():
            if k in remove_dict:
                col_table_dict_remove[remove_dict[k]] = v
                enable_table += v


        col_table_dict = col_table_dict_remove

        col_set_type_remove = np.zeros((len(col_set_iter), 4))
        for k, v in remove_dict.items():
            if v < len(col_set_iter_remove):
                col_set_type_remove[v] = col_set_type[k]

        col_set_type = col_set_type_remove


        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, [" ".join(x) for x in question_arg],
                                                                [" ".join(x) for x in col_set_iter])


        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)


        if len(input_ids1) > 100:
            continue

        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, [" ".join(x) for x in question_arg],
                                                                [" ".join(x) for x in table_names])
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        if len(input_ids1) > 100:
            continue




        example = Example(
            src_sent=[" ".join(x) for x in question_arg],
            # src_sent=origin_sql,
            col_num=len(col_set_iter),
            vis_seq=(sql['question'], col_set_iter, sql['query']),
            tab_cols=col_set_iter,
            tgt_actions=rule_label,
            sql=sql['query'],
            one_hot_type=one_hot_type,
            col_hot_type=col_set_type,
            schema_len=table['schema_len'],
            table_names=table_names,
            table_len=len(table_names),
            col_table_dict=col_table_dict,
            table_set_type=table_set_type,
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            is_sketch=is_sketch,
            pos_tags=pos_tags,
            dependency_adjacency=dependency_graph_adjacency_matrix,
            entities=entities,
            sketch_adjacency_matrix=None

        )
        example.remove_dict_reverse = remove_dict_reverse
        example.sql_json=copy.deepcopy(sql)
        examples.append(example)
    if is_train:
        examples.sort(key=lambda e: -len(e.src_sent))
    # quit()
        return examples
    else:
        return examples, col_org_seq, schema_seq

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result


def epoch_train(model, optimizer, batch_size, sql_data, table_data,
                schemas, args, is_train=True, is_sketch=False,
                has_pos_tags=False,
                is_encode_dependency=False,
                is_encode_entity=False, epoch=0):
    model.train()
    # shuffe
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, schemas,
                                is_sketch=is_sketch,
                                has_pos_tags=has_pos_tags,
                                is_encode_dependency=is_encode_dependency,
                                is_encode_entity=is_encode_entity, move_place=epoch//3)
        optimizer.zero_grad()

        score = model.forward(examples)
        loss_sketch = -score[0]
        loss_lf = -score[1]
        #
        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)

        # print(loss_sketch, loss_lf)
        #
        loss = loss_lf + loss_sketch


        # TODO: what is the sup_attention?
        loss.backward()
        if args.clip_grad > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        # some records
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        st = ed
    return cum_loss / len(sql_data)


def is_validAction(actions):
    for id_x, action in enumerate(actions):
        if type(action) == C:
            if id_x == len(actions) - 1:
                return False
            if type(actions[id_x + 1]) != T:
                return False
    return True


def calc_beam_acc(beam_search_result, example):
    results = list()
    beam_result = False
    truth = " ".join([str(x) for x in example.tgt_actions]).strip()
    for bs in beam_search_result:
        pred = " ".join([str(x) for x in bs.actions]).strip()
        if truth == pred:
            results.append(True)
            beam_result = True
        else:
            results.append(False)
    return results[0], beam_result


def epoch_acc(model, batch_size, sql_data, table_data, schemas, beam_size=3, is_sketch=False,
              has_pos_tags=False,
              is_encode_dependency=False,
              is_encode_entity=False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    one_sketch_num = 0.0
    total_sql = 0
    right_result = []
    wrong_result = []
    beam_wrong_result = list()

    best_correct = 0
    beam_correct = 0

    sel_num = []
    sel_col = []
    agg_col = []
    ori_col = []
    table_col = []
    shema_query = []
    json_datas = []
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples, ori_col_, shema_query_ = to_batch_seq(sql_data, table_data, perm, st, ed, schemas,
                                                        is_train=False,
                                                        is_sketch=is_sketch,
                                                        has_pos_tags=has_pos_tags,
                                                        is_encode_dependency=is_encode_dependency,
                                                        is_encode_entity=is_encode_entity)
        ori_col.extend(ori_col_)
        shema_query.extend(shema_query_)
        for example in examples:


            results_all = model.parse(example, beam_size=beam_size)

            try:
                results = results_all[0]
                sketch_actions = " ".join(str(x) for x in results_all[1])
                list_preds = []
                for x in results[0].actions:
                    if type(x) == C:
                        x.id_c = example.remove_dict_reverse[x.id_c]
                pred = " ".join([str(x) for x in results[0].actions])
                for x in results:
                    list_preds.append(" ".join(str(x.actions)))
            except Exception as e:
                pred = ""
                sketch_actions = ""

            simple_json = example.sql_json
            simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred
            json_datas.append(simple_json)

            example.sql_json['model_result'] = pred
            # print(example.sql_json)

            # example.sql_json['fusion_results'] = list_preds


            # json_datas.append(example.sql_json)

            if len(results) > 0:
                pred = []
                x_id = 0
                while x_id < len(results[0].actions):
                    pred.append(results[0].actions[x_id])
                    if type(results[0].actions[x_id]) == C and results[0].actions[x_id].id_c == 0:

                        x_id += 1
                    x_id += 1
                pred = " ".join([str(x) for x in pred])
            else:
                pred = " "




            glod = []
            x_id = 0
            while x_id < len(example.tgt_actions):
                if type(example.tgt_actions[x_id]) == C:
                    example.tgt_actions[x_id].id_c = example.remove_dict_reverse[example.tgt_actions[x_id].id_c]
                glod.append(example.tgt_actions[x_id])
                if type(example.tgt_actions[x_id]) == C and example.tgt_actions[x_id].id_c == 0:
                    x_id += 1
                x_id += 1
            glod = " ".join([str(x) for x in glod])

            sketch_glod = " ".join([str(x) for x in example.sketch])

            # glod = " ".join([str(x) for x in example.tgt_actions])
            src_str = " ".join([str(x) for x in example.src_sent])
            # print(sketch_glod)
            # print(sketch_actions)
            # print('======')
            if sketch_glod == sketch_actions:
                one_sketch_num += 1
            if pred == glod:
                one_acc_num += 1
                # right_result.append((pred, glod, src_str, example.sql))
            else:
                # wrong_result.append((pred, glod, src_str, example.sql))
                pass
                # print(glod)
                # for re in results:
                #     for action in re.actions:
                #         print(action, ':',  action.score.data.cpu().numpy(),' ', end="")
                #     print('')
                #     print('=====')

            glod = " ".join([str(x) for x in example.tgt_actions]).strip()
            if len(results) > 0:
                pred = " ".join([str(x) for x in results[0].actions]).strip()
                _best_correct, _beam_correct = calc_beam_acc(results, example)
            else:
                pred = ""
                _best_correct, _beam_correct = False, False
            if _beam_correct:
                beam_correct += 1
            else:
                preds = [" ".join([str(x) for x in r.actions]) for r in results]
                preds.append(glod)
                preds.append(src_str)
                preds.append(example.sql)
                beam_wrong_result.append(preds)
            if _best_correct:
                best_correct += 1
                right_result.append((pred, glod, src_str, example.sql))
            else:
                wrong_result.append((pred, glod, src_str, example.sql))

            total_sql += 1

        st = ed
    # # print('total acc is : ', one_acc_num / total_sql)
    with open('lf_predict.json', 'w') as f:
        json.dump(json_datas, f)
    print('sketch acc is ', one_sketch_num / total_sql)
    print(best_correct / total_sql, beam_correct / total_sql)
    # quit()
    return best_correct / total_sql, beam_correct / total_sql, (right_result, wrong_result, beam_wrong_result), \
           ((sel_num, sel_col, agg_col, table_col), ori_col, shema_query)


def find_col_id(cols, cols_id, action_c, action_t):
    for ids, (a, b) in enumerate(zip(cols, cols_id)):
        if a == action_c and b == action_t:
            return ids
        if action_c == '*':
            return 0
    print(cols, cols_id)
    print(action_c, action_t)
    raise NotImplementedError("find_col_id wrong")

def gen_from(candidate_tables, schema):
    if len(candidate_tables) <= 1:
        if len(candidate_tables) == 1:
            ret = "from {}".format(schema["table_names_original"][list(candidate_tables)[0]])
        else:
            ret = "from {}".format(schema["table_names_original"][0])
        # TODO: temporarily settings for select count(*)
        return {}, ret
    # print("candidate:{}".format(candidate_tables))
    table_alias_dict = {}
    uf_dict = {}
    for t in candidate_tables:
        uf_dict[t] = -1
    idx = 1
    graph = collections.defaultdict(list)
    for acol, bcol in schema["foreign_keys"]:
        t1 = schema["column_names"][acol][0]
        t2 = schema["column_names"][bcol][0]
        graph[t1].append((t2, (acol, bcol)))
        graph[t2].append((t1, (bcol, acol)))
    candidate_tables = list(candidate_tables)
    start = candidate_tables[0]
    table_alias_dict[start] = idx
    idx += 1
    ret = "from {} as T1".format(schema["table_names_original"][start])
    try:
        for end in candidate_tables[1:]:
            if end in table_alias_dict:
                continue
            path = find_shortest_path(start, end, graph)
            prev_table = start
            if not path:
                table_alias_dict[end] = idx
                idx += 1
                ret = "{} join {} as T{}".format(ret, schema["table_names_original"][end],
                                                 table_alias_dict[end],
                                                 )
                continue
            for node, (acol, bcol) in path:
                if node in table_alias_dict:
                    prev_table = node
                    continue
                table_alias_dict[node] = idx
                idx += 1
                ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(ret, schema["table_names_original"][node],
                                                                    table_alias_dict[node],
                                                                    table_alias_dict[prev_table],
                                                                    schema["column_names_original"][acol][1],
                                                                    table_alias_dict[node],
                                                                    schema["column_names_original"][bcol][1])
                prev_table = node
    except Exception as e:
        print(e)
        print("db:{}".format(schema["db_id"]))
        # print(table["db_id"])
        return table_alias_dict, ret
    return table_alias_dict, ret

def find_shortest_path(start, end, graph):
    stack = [[start, []]]
    visited = set()
    while len(stack) > 0:
        ele, history = stack.pop()
        if ele == end:
            return history
        for node in graph[ele]:
            if node[0] not in visited:
                stack.append((node[0], history + [(node[0], node[1])]))
                visited.add(node[0])

def transformer_data(actions):
    sel_act = {}
    for aid, action in enumerate(actions):
        if type(action) == C:
            sel_act[action.id_c] = sel_act.get(action.id_c, []) + [actions[aid-1].id_c]
    keys = []
    values = []
    for k, v in sel_act.items():
        keys.append(k)
        values.append(v)
    return (len(sel_act), keys, values)


def gen_sql(score, col_org, schema_seq):
    sel_num_cols, sel_col, agg_preds_gen_all, table_col = score
    B = len(sel_num_cols)
    ret_sqls = []

    for b in range(B):
        cur_cols = col_org[b]
        schema = schema_seq[b]
        cur_query = {}
        cur_tables = collections.defaultdict(list)

        cur_sel = []

        # ------------get sel predict
        cur_query['sel_num'] = sel_num_cols[b]
        cur_query['sel'] = sel_col[b]
        agg_preds_gen = agg_preds_gen_all[b]

        cur_sel.append("select")
        for i, cid in enumerate(cur_query['sel']):
            aggs = agg_preds_gen[i]
            agg_num = len(aggs)
            for j, gix in enumerate(aggs):
                if gix == 0:
                    cur_sel.append([cid, cur_cols[cid][1]])
                    cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                else:
                    cur_sel.append(AGG_OPS[gix])
                    cur_sel.append("(")
                    cur_sel.append([cid, cur_cols[cid][1]])
                    cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                    cur_sel.append(")")

                if j < agg_num - 1:
                    cur_sel.append(",")
            if i < sel_num_cols[b] - 1:
                cur_sel.append(",")
        # print(cur_tables)
        # print(cur_tables.keys())
        if -1 in cur_tables.keys():
            del cur_tables[-1]
            assert table_col[b] != -1
            cur_tables[table_col[b]].append([0, '*'])
        # print(cur_sel)
        # quit()
        # print(cur_tables)
        table_alias_dict, ret = gen_from(list(cur_tables.keys()), schema)
        # print(table_alias_dict)
        if len(table_alias_dict) > 0:
            col_map = {}
            for tid, aid in table_alias_dict.items():
                for cid, col in cur_tables[tid]:
                    col_map[cid] = "t" + str(aid) + "." + col
            new_sel = []
            for s in cur_sel:
                if isinstance(s, list):
                    if s[0] == 0:
                        new_sel.append("*")
                    elif s[0] in col_map:
                        new_sel.append(col_map[s[0]])
                else:
                    new_sel.append(s)

            cur_sql = new_sel + [ret]

        else:
            cur_sql = []
            cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_sel])
            if len(cur_tables.keys()) == 0:
                cur_tables[0] = []
            cur_sql.extend(["from", schema["table_names_original"][list(cur_tables.keys())[0]]])

        sql_str = " ".join(cur_sql)
        # print sql_str
        # quit()
        ret_sqls.append(sql_str)

    return ret_sqls


def build_vocab(dataset, vocab_path):

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset, use_small=False)

    voc = []
    table_voc = list()

    for sql in sql_data:
        for v in sql['col_iter']:
            for z in v:
                voc.append(z)
                table_voc.append(z)
        for t in sql['table_names']:
            table_voc.append(t)

    for sql in val_sql_data:
        for v in sql['col_iter']:
            for z in v:
                voc.append(z)
                table_voc.append(z)
        for t in sql['table_names']:
            table_voc.append(t)

    for sql in sql_data:
        for v in sql['stanford_tokenized']:
            voc.append(v)
    for sql in val_sql_data:
        for v in sql['stanford_tokenized']:
            voc.append(v)

    vocab = pickle.load(open(vocab_path, 'rb'))
    table_vocab = copy.deepcopy(vocab)

    vocab.source = VocabEntry.from_corpus(voc, 100000, 3)
    table_vocab.source = VocabEntry.from_corpus(table_voc, 100000, 1)

    path, _ = os.path.split(vocab_path)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(os.path.join(path, 'table_vocab.bin'), 'wb') as f:
        pickle.dump(table_vocab, f)



def normalize_adjacency_matrix(matrix):
    matrix += np.identity(len(matrix))
    diagonal = np.sum(matrix, axis=-1)
    diagonal = 1 / np.sqrt(diagonal)
    d = np.diag(diagonal)
    return np.matmul(np.matmul(d, matrix), d)


def build_sketch_adjacency_matrix(rule_labels):
    """
    :param rule_labels: Array of Action Instance [Root1(3), Root(0)]
    :return:
    """
    lf.build_tree(copy.copy(rule_labels))
    sketch_adjacency_matrix = lf.build_adjacency_matrix(rule_labels, symmetry=True)
    return normalize_adjacency_matrix(sketch_adjacency_matrix)

def get_parent_actions(rule_labels, sketch):
    parent_actions = list()
    parent_actions_idx = list()
    for idx, t_action in enumerate(rule_labels):
        if idx > 0 and rule_labels[idx - 1] == t_action.parent:
            parent_actions.append(None)
            parent_actions_idx.append(None)
        else:
            parent_actions.append(t_action.parent)
            parent_actions_idx.append(sketch.index(t_action.parent) if t_action.parent is not None else None)
    return parent_actions, parent_actions_idx

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, help='Path to dataset')
    arg_parser.add_argument('--vocab', type=str, help='Path to vocab')
    args = arg_parser.parse_args()
    build_vocab(args.dataset, args.vocab)
