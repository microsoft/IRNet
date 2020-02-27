# coding=utf8

import os
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import nn_utils
from hypothesis import Hypothesis, ActionInfo
from dataset import Batch
from pointer_net import PointerNet
from nn_utils import MultiHeadAttention, EncoderLayer, get_attn_key_pad_mask, get_non_pad_mask
from rule import define_rule
import copy
from utils import build_sketch_adjacency_matrix, get_parent_actions
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from bert_utils import get_wemb_bert, encode_hpu


device = torch.device("cuda")
class Args:
    def __init__(self):
        pass
bert_config = Args()
bert_config.hidden_size = 768
bert_config.num_hidden_layers = 12


class Seq2Tree(nn.Module):
    def __init__(self, args, vocab, grammar, table_vocab):
        super(Seq2Tree, self).__init__()
        self.vocab = vocab
        self.args = args
        self.grammar = grammar
        self.table_vocab = table_vocab
        self.is_encode_sketch = args.encode_sketch
        self.is_parent_feeding = args.parent_feeding
        self.use_sketch_history = args.sketch_history
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features
        self.use_stanford_tokenized = args.stanford_tokenized

        self.col_enc_n = nn.LSTM(input_size=768, hidden_size=int(768 / 2),
                                 num_layers=1, batch_first=True,
                                 dropout=0.2, bidirectional=True).cuda()

        self.table_enc_n = nn.LSTM(input_size=768, hidden_size=int(768 / 2),
                                   num_layers=1, batch_first=True,
                                   dropout=0.2, bidirectional=True).cuda()


        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        if args.lstm == 'lstm':

            input_dim = args.action_embed_size  # previous action
            # frontier info

            input_dim += args.att_vec_size  # input feeding

            input_dim += args.type_embed_size  # pre type embedding
            lf_input_dim = input_dim
            if self.is_parent_feeding:
                lf_input_dim += args.action_embed_size
            if self.use_sketch_history:
                lf_input_dim += args.hidden_size

            self.lf_decoder_lstm = nn.LSTMCell(lf_input_dim, args.hidden_size)

            self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        self.bertModel = BertModel.from_pretrained(args.bert_model)
        self.bertModel.to(device)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.encoder_dim , args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.encoder_dim, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.encoder_dim + args.hidden_size , args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.encoder_dim + args.hidden_size, args.att_vec_size, bias=False)

        self.sketch_begin_vec = Variable(self.new_tensor(self.sketch_decoder_lstm.input_size))
        self.lf_begin_vec = Variable(self.new_tensor(self.lf_decoder_lstm.input_size))


        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)


        self.col_type = nn.Linear(4, 768)
        self.sketch_encoder = nn.LSTM(args.action_embed_size, int(args.action_embed_size / 2), bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)
        self.fuse_sketch_encodings = nn.Sequential(
            nn.Linear(args.action_embed_size * 2, args.action_embed_size, bias=True), nn.ReLU())

        # args.N_embed_size
        self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.action_embed_size)

        # args.readout
        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity
        # # args.embed_size args.att_vec_size
        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.q_att = nn.Linear(args.hidden_size, args.embed_size)

        self.column_rnn_input = nn.Linear(768, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(768, args.action_embed_size, bias=False)

        # self.col_appear = nn.Sequential(nn.Sigmoid(), nn.Linear(1, 1))
        self.col_appear = nn.Linear(1, 1, bias=False)

        # self.q_num_att = nn.Linear(args.embed_size, args.embed_size, bias=False)

        self.col_weight = nn.Linear(3, 1, bias=False)

        if self.use_sentence_features:
            self.q_type_project = nn.Linear(6, 6)
        else:
            self.q_type_project = nn.Sequential(nn.Linear(3, 3), nn.ReLU())


        self.t_type_project = nn.Linear(3, args.col_embed_size)

        self.t_attention_out = nn.Linear(3, 1, bias=False)

        self.col_emd_trans = nn.Linear(args.hidden_size, args.col_embed_size)
        self.tab_emd_trans = nn.Linear(args.hidden_size, args.col_embed_size)

        # self.table_project = nn.Linear(5 + args.embed_size, args.embed_size)
        self.col_attention_out = nn.Linear(3, 1)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, 768, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, 768, attention_type=args.column_att)

        heads = 4
        self.historyAware = EncoderLayer(self.sketch_decoder_lstm.input_size, 128, heads,
                                         self.sketch_decoder_lstm.input_size // heads,
                                         self.sketch_decoder_lstm.input_size // heads, dropout=0.2)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        print('Encode Sketch: ', True if self.is_encode_sketch else False)
        print('Parent Feeding: ', True if self.is_parent_feeding else False)
        print('Use Sketch History: ', True if self.use_sketch_history else False)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)
        print('Use Sentence Features: ', True if self.use_sentence_features else False)
        print('Use Stanford Tokenized: ', True if self.use_stanford_tokenized else False)

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        # for the formulate ~s_t = tanh(W_c[C_t : S_t])
        # att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def forward(self, examples):
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, self.vocab, cuda=self.args.cuda, table_vocab=self.table_vocab)

        table_appear_mask = batch.table_appear_mask
        schema_appear_mask = batch.schema_appear_mask
        src_encodings, table_encoding, src_mask = self.encoding_src_col(batch, batch.src_sents_word, batch.table_sents_word,
                                                                 self.col_enc_n)

        _, schema_encoding, _ = self.encoding_src_col(batch,  batch.src_sents_word, batch.schema_sents_word,
                                                             self.table_enc_n)

        src_encodings = self.dropout(src_encodings)


        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = (Variable(self.new_tensor(len(examples), args.hidden_size).zero_()), Variable(self.new_tensor(len(examples), args.hidden_size).zero_()))
        h_tm1 = dec_init_vec
        action_probs = [[] for _ in examples]

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        parent_zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_(), requires_grad=False)
        zero_type_embed = Variable(self.new_tensor(args.type_embed_size).zero_())

        sketch_attention_history = list()

        for t in range(batch.max_sketch_num):
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_(),
                             requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        else:
                            print(action_tm1, 'only for sketch')
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)


            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            sketch_attention_history.append(att_t)

            # get the Root possibility
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        sketch_attention_history = torch.stack(sketch_attention_history)

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = F.relu(self.col_type(col_type))


        table_encoding = table_encoding + col_type_var
        # col_pred_mask = batch.pred_col_mask

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

        if self.is_encode_sketch:
            sketch_encodings = self.encode_sketch(batch.sketch_vars, batch.sketch_len_vars, batch.sketch_len_mask_vars,
                                                  batch.sketch_adjacency_matrix_var)
        else:
            sketch_encodings = None

        sketch_steps = [0 for i in range(len(examples))]
        h_tm1 = dec_init_vec
        sketch_zero_history = Variable(self.new_tensor(self.args.hidden_size).zero_(), requires_grad=False)

        for t in range(batch.max_action_num):
            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0).repeat(len(batch), 1)
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []
                parent_action_embeds = []
                sketch_history = []
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order,
                                                ]:
                            if self.is_encode_sketch:
                                a_tm1_embed = sketch_encodings[e_id][sketch_steps[e_id], :]
                            else:
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            _history = sketch_attention_history[sketch_steps[e_id]][e_id]
                            sketch_steps[e_id] += 1
                        else:
                            _history = sketch_zero_history
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(table_encoding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.table_rnn_input(schema_encoding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                        parent_action = example.parent_actions[t]
                        if parent_action is None:
                            parent_action_embed = parent_zero_action_embed
                        else:
                            parent_action_embed = self.production_embed.weight[
                                self.grammar.prod2id[parent_action.production]]

                    else:
                        _history = sketch_zero_history
                        a_tm1_embed = zero_action_embed
                        parent_action_embed = parent_zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)
                    parent_action_embeds.append(parent_action_embed)
                    sketch_history.append(_history)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                parent_action_embeds = torch.stack(parent_action_embeds)
                sketch_history = torch.stack(sketch_history)

                inputs = [a_tm1_embeds]
                if self.is_parent_feeding:
                    inputs.append(parent_action_embeds)

                if self.use_sketch_history:
                    inputs.append(sketch_history)

                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=table_encoding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=table_encoding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = self.column_pointer_net(src_encodings=table_encoding, query_vec=att_t.unsqueeze(0),
                #                                   src_token_mask=batch.table_token_mask)
            else:
                weights = self.column_pointer_net(src_encodings=table_encoding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)
            # weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=batch.table_token_mask)
            weights.data.masked_fill_(batch.table_token_mask.bool(), -float('inf'))
            # weights.data.masked_fill_(col_pred_mask.bool(), -float('inf'))

            column_attention_weights = F.softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=schema_encoding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
            # table_weights = self.table_pointer_net(src_encodings=schema_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.softmax(table_weights, dim=-1)

            # now we should get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.T):
                        schema_appear_mask[e_id, action_t.id_c] = 1
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        # act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        # action_probs[e_id].append(act_prob_t_i)
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        return [sketch_prob_var, lf_prob_var]

    def parse(self, examples, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        args = self.args
        # now should implement the examples
        batch = Batch([examples], self.grammar, self.vocab, cuda=self.args.cuda, table_vocab=self.table_vocab)

        src_encodings, table_encoding, src_mask = self.encoding_src_col(batch, batch.src_sents_word, batch.table_sents_word,
                                                                 self.col_enc_n)

        _, schema_encoding, _ = self.encoding_src_col(batch, batch.src_sents_word, batch.schema_sents_word,
                                                             self.table_enc_n)

        src_encodings = self.dropout(src_encodings)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = (Variable(self.new_tensor(len(batch), self.args.hidden_size).zero_()),
                        Variable(self.new_tensor(len(batch), self.args.hidden_size).zero_()))

        h_tm1 = dec_init_vec

        t = 0
        hypotheses = [Hypothesis(is_sketch=True)]
        completed_hypotheses = []
        hyp_states = [[]]
        history_hids = []
        history_states = []
        while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # expand value
            exp_sketch_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                            src_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num,
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           1),
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           2))

            if t == 0:
                # x = self.sketch_begin_vec.unsqueeze(0)
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(hypotheses):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(hypotheses):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_sketch_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstm,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)
            new_hyp_meta = []

            for hyp_id, hyp in enumerate(hypotheses):
                action_class = hyp.get_availableClass()
                hyp.sketch_attention_history.append(att_t[hyp_id])
                if action_class in [define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError('No right action class')

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = hypotheses[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                if t > 0:
                    # TODO: implement the parent feeding
                    pass

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                history_hids.append(att_tm1)

                hypotheses = new_hypotheses
                t += 1
            else:
                break

        # now get the sketch result
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        if len(completed_hypotheses) == 0:
            return [[], []]
        sketch_actions = completed_hypotheses[0].actions
        sketch_attention_history = torch.stack(completed_hypotheses[0].sketch_attention_history)
        # sketch_actions = examples.sketch

        padding_sketch = self.padding_sketch(sketch_actions)
        parent_actions, parent_actions_idx = get_parent_actions(padding_sketch, sketch_actions)
        # padding_sketch = self.padding_sketch(examples.sketch)
        # padding_sketch = examples.tgt_actions

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = F.relu(self.col_type(col_type))

        table_encoding = table_encoding + col_type_var
        # col_pred_mask = batch.pred_col_mask

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        sketch_zero_history = Variable(self.new_tensor(self.args.hidden_size).zero_(), requires_grad=False)
        parent_zero_action_embed = Variable(self.new_tensor(self.args.action_embed_size).zero_(), requires_grad=False)
        t = 0
        hypotheses = [Hypothesis(is_sketch=False)]
        completed_hypotheses = []
        hyp_states = [[]]
        history_hids = []

        if self.is_encode_sketch:
            matrix = self.get_sketch_adjacency_matrix(padding_sketch)
            sketch_vars = torch.as_tensor([[self.grammar.prod2id[rule.production] for rule in sketch_actions]],
                                          dtype=torch.long)
            if self.cuda:
                sketch_vars = sketch_vars.cuda()
            sketch_len_vars = torch.as_tensor([len(sketch_actions)], dtype=torch.long)
            if self.cuda:
                sketch_len_vars = sketch_len_vars.cuda()
            sketch_len_mask_vars = nn_utils.length_array_to_mask_tensor([len(sketch_actions)], self.cuda)
            sketch_encodings = self.encode_sketch(sketch_vars, sketch_len_vars, sketch_len_mask_vars,
                                                  matrix.unsqueeze(0))
        else:
            sketch_encodings = None

        while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # expand value
            exp_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num,
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         1),
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         2))
            exp_table_embedding = table_encoding.expand(hyp_num, table_encoding.size(1),
                                                         table_encoding.size(2))

            exp_schema_embedding = schema_encoding.expand(hyp_num, schema_encoding.size(1),
                                                          schema_encoding.size(2))


            # exp_col_pred_mask = col_pred_mask.expand(hyp_num, col_pred_mask.size(1))

            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(hypotheses):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0)
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                parent_action_embeds = list()
                sketch_history = list()
                for e_id, hyp in enumerate(hypotheses):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        if self.is_encode_sketch:
                            a_tm1_embed = sketch_encodings[0][hyp.sketch_step, :]
                        else:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        _history = sketch_attention_history[hyp.sketch_step]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        _history = sketch_zero_history
                        a_tm1_embed = self.column_rnn_input(table_encoding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.T):
                        _history = sketch_zero_history
                        a_tm1_embed = self.table_rnn_input(schema_encoding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A):
                        _history = sketch_zero_history
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    parent_action = parent_actions[t]
                    if parent_action is None:
                        parent_action_embed = parent_zero_action_embed
                    else:
                        parent_action_embed = self.production_embed.weight[
                            self.grammar.prod2id[parent_action.production]]

                    a_tm1_embeds.append(a_tm1_embed)
                    parent_action_embeds.append(parent_action_embed)
                    sketch_history.append(_history)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                parent_action_embeds = torch.stack(parent_action_embeds)
                sketch_history = torch.stack(sketch_history)

                inputs = [a_tm1_embeds]
                if self.is_parent_feeding:
                    inputs.append(parent_action_embeds)

                if self.use_sketch_history:
                    inputs.append(sketch_history)

                for e_id, hyp in enumerate(hypotheses):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                #                                   src_token_mask=batch.table_token_mask)
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            # weights.data.masked_fill_(exp_col_pred_mask.bool(), -float('inf'))

            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.log_softmax(table_weights, dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(hypotheses):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents_var[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.schema_sents_var[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = hypotheses[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                # if new_hyp.is_valid is False:
                #     continue

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                history_hids.append(att_tm1)

                hypotheses = new_hypotheses
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return [completed_hypotheses, sketch_actions]

    def encode_sketch(self, sketch_vars, sketch_length, sketch_len_mask, sketch_adjacency_vars=None):
        """
        Reference: https://discuss.pytorch.org/t/rnns-sorting-operations-autograd-safe/1461/5
        Encode sketch with BiLSTM
        :param sketch_adjacency_vars:
        :param sketch_vars:
        :param sketch_len_var:
        :return:
        """
        _lengths, perm_index = sketch_length.sort(0, descending=True)
        _, initial_index = perm_index.sort(0, descending=False)
        sketch_rule_embeds = self.production_embed(sketch_vars)
        sketch_rule_embeds.masked_fill_(sketch_len_mask.unsqueeze(2).repeat(1, 1, sketch_rule_embeds.shape[-1]).bool(), 0.0)
        _embeds = sketch_rule_embeds[perm_index]
        packed_src_token_embed = pack_padded_sequence(_embeds, list(_lengths.data), batch_first=True)
        sketch_encodings, _ = self.sketch_encoder(packed_src_token_embed)
        sketch_encodings, _ = pad_packed_sequence(sketch_encodings, batch_first=True)
        sketch_encodings = sketch_encodings[initial_index]
        # return self.sketch_encoder(sketch_rule_embeds, sketch_adjacency_vars)
        # sketch_encodings = self.fuse_sketch_encodings(torch.cat([sketch_encodings, sketch_rule_embeds], -1))
        return sketch_encodings

    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))

        return padding_result

    def get_sketch_adjacency_matrix(self, sketch):
        matrix = build_sketch_adjacency_matrix(sketch)
        matrix_tensor = torch.as_tensor(matrix, dtype=torch.float)
        if self.cuda:
            return matrix_tensor.cuda()
        return matrix_tensor

    def encoding_src_col(self, batch, src_sents_word, table_sents_word, enc_n):
        wemb_n, wemb_h, l_n, n_hs, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx = get_wemb_bert(bert_config, self.bertModel, self.tokenizer,src_sents_word, table_sents_word, 100,
                                                         num_out_layers_n=1, num_out_layers_h=1)
        emb = encode_hpu(enc_n, wemb_h, l_hpu=l_hpu, l_hs=l_hs)
        return wemb_n, emb, batch.len_appear_mask(n_hs)


    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
