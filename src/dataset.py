# coding=utf8

import copy
import torch
import numpy as np
import nn_utils
from rule import define_rule


class Example:
    """

    """
    def __init__(self, src_sent, tgt_actions, vis_seq=None, tab_cols=None, col_num=None, sql=None,
                 one_hot_type=None, col_hot_type=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None, cols_id=None, cols_set=None,
                 table_set_type=None, table_col_name=None, table_col_len=None, is_sketch=False, pos_tags=None,
                 dependency_adjacency=None, entities=None, col_pred=None, sketch_adjacency_matrix=None, sketch=None
                 ):
        self.src_sent = src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.sql = sql
        self.one_hot_type=one_hot_type
        self.col_hot_type = col_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.cols_id = cols_id
        self.cols_set = cols_set
        self.table_set_type = table_set_type
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.pos_tags = pos_tags
        self.entities = entities
        self.col_pred = col_pred
        self.dependency_adjacency = dependency_adjacency
        self.sketch_adjacency_matrix = sketch_adjacency_matrix

        self.truth_actions = copy.deepcopy(tgt_actions)
        if is_sketch:
            self.tgt_actions = list()
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A):
                    continue
                self.tgt_actions.append(ta)
        else:
            self.tgt_actions = list()
            for ta in self.truth_actions:
                self.tgt_actions.append(ta)
        if sketch:
            self.sketch = sketch
        else:
            self.sketch = list()
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A):
                    continue
                self.sketch.append(ta)

        self.parent_actions = list()
        self.parent_actions_idx = list()
        for idx, t_action in enumerate(self.tgt_actions):
            if idx > 0 and self.tgt_actions[idx - 1] == t_action.parent:
                self.parent_actions.append(None)
                self.parent_actions_idx.append(None)
            else:
                self.parent_actions.append(t_action.parent)
                self.parent_actions_idx.append(
                    self.sketch.index(t_action.parent) if t_action.parent is not None else None)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar, vocab, cuda=False, table_vocab=None, pos_tags_dict=None):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]

        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.schema_len = [e.schema_len for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_set_type = [e.table_set_type for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.sketch_adjacency_matrix = [e.sketch_adjacency_matrix for e in examples]
        self.sketches = [e.sketch for e in examples]


        self.pos_tags = [e.pos_tags for e in examples]
        self.entities = [e.entities for e in examples]
        self.dependency_adjacency = [e.dependency_adjacency for e in examples]

        self.grammar = grammar
        self.vocab = vocab
        self.cuda = cuda
        self.pos_tags_dict = pos_tags_dict

        if table_vocab is None:
            self.table_vocab = vocab
        else:
            self.table_vocab = table_vocab

    def __len__(self):
        return len(self.examples)

    def get_frontier_field_idx(self, t):
        pass

    def get_frontier_prod_idx(self, t):
        pass

    def get_frontier_field_type_idx(self, t):
        pass

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    def len_appear_mask(self, length):
        return nn_utils.length_array_to_mask_tensor(length, cuda=self.cuda)

    @property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def schema_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def pred_col_mask(self):
        return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def table_col_var(self):
        return nn_utils.to_input_variable(self.table_col_name, self.table_vocab.source, cuda=self.cuda)

    @cached_property
    def schema_sents_var(self):
        return nn_utils.to_input_variable(self.table_names, self.table_vocab.source, cuda=self.cuda)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_names_var(self):
        return nn_utils.to_input_variable(self.table_names, self.table_vocab.source, cuda=self.cuda)

    @cached_property
    def table_sents_var(self):
        return nn_utils.to_input_variable(self.table_sents, self.table_vocab.source,
                                          cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)


    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=self.table_sents_var)

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          cuda=self.cuda)
    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def src_position(self):
        return nn_utils.length_position_tensor(self.src_sents_len,
                                                cuda=self.cuda)

    @cached_property
    def pos_tags_var(self):
        if self.pos_tags_dict is None:
            return None
        lengths = [len(ts) for ts in self.pos_tags]
        max_length = max(lengths)
        pos_tag_ids = list()
        for ts in self.pos_tags:
            var = list()
            for t in ts:
                var.append(self.pos_tags_dict[t])
            for i in range(max_length - len(ts)):
                var.append(len(self.pos_tags_dict))
            pos_tag_ids.append(var)
        pos_tag_ids = np.array(pos_tag_ids)
        pos_tag_ids = torch.as_tensor(pos_tag_ids, dtype=torch.long)
        if self.cuda:
            return pos_tag_ids.cuda()
        return pos_tag_ids

    @cached_property
    def entity_var(self):
        if self.entities[0] is None:
            return None
        entity_map = {
            'O': 0,
            'B': 1,
            'I': 2,
        }
        lengths = [len(ts) for ts in self.entities]
        max_length = max(lengths)
        entity_ids = list()
        for ts in self.entities:
            var = list()
            for t in ts:
                var.append(entity_map[t])
            for i in range(max_length - len(ts)):
                var.append(len(entity_map))
            entity_ids.append(var)
        entity_ids = np.array(entity_ids)
        entity_ids = torch.as_tensor(entity_ids, dtype=torch.long)
        if self.cuda:
            return entity_ids.cuda()
        return entity_ids

    @cached_property
    def dependency_adjacency_var(self):
        if self.dependency_adjacency[0] is None:
            return None
        return nn_utils.pad_matrix(self.dependency_adjacency, self.cuda)

    @cached_property
    def sketch_adjacency_matrix_var(self):
        if self.sketch_adjacency_matrix[0] is None:
            return None
        return nn_utils.pad_matrix(self.sketch_adjacency_matrix, self.cuda)

    @cached_property
    def sketch_vars(self):
        if self.sketches[0] is None:
            return None
        max_length = max([len(s) for s in self.sketches])
        vars = list()
        for sketch in self.sketches:
            v = list()
            for rule in sketch:
                v.append(self.grammar.prod2id[rule.production])

            #FIXME: why padding as 0?
            v += [0 for i in range(max_length - len(sketch))]
            vars.append(v)
        vars = torch.as_tensor(vars, dtype=torch.long)
        if self.cuda:
            return vars.cuda()
        return vars

    @cached_property
    def sketch_len(self):
        if self.sketches[0] is None:
            return None
        return [len(s) for s in self.sketches]

    @cached_property
    def sketch_len_mask(self):
        if self.sketches[0] is None:
            return None
        max_length = max([len(s) for s in self.sketches])

        mask = np.ones((len(self.sketches), max_length), dtype=np.uint8)
        for i, seq in enumerate(self.sketches):
            mask[i][:len(seq)] = 0

        mask = torch.ByteTensor(mask)
        return mask.cuda() if self.cuda else mask



