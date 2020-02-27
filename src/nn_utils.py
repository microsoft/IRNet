# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from six.moves import xrange


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with tree layers'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask.bool(), -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


def length_array_to_mask_tensor(length_array, cuda=False, value=None):
    # if bert is True:
    #     max_len = max(length_array) + 2
    # else:
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    if value != None:
        for b_id in range(len(value)):
            for c_id, c in enumerate(value[b_id]):
                if value[b_id][c_id] == [3]:
                    mask[b_id][c_id] = 1

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def table_dict_to_mask_tensor(length_array, table_dict, cuda=False ):
    max_len = max(length_array)
    batch_size = len(table_dict)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, ta_val in enumerate(table_dict):
        for tt in ta_val:
            mask[i][tt] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def length_position_tensor(length_array, cuda=False, value=None):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for b_id in range(batch_size):
        for len_c in range(length_array[b_id]):
            mask[b_id][len_c] = len_c + 1

    mask = torch.LongTensor(mask)
    return mask.cuda() if cuda else mask


def appear_to_mask_tensor(length_array, cuda=False, value=None):
    max_len = max(length_array)
    batch_size = len(length_array)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    return mask

def pred_col_mask(value, max_len):
    max_len = max(max_len)
    batch_size = len(value)
    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for v_ind, v_val in enumerate(value):
        for v in v_val:
            mask[v_ind][v] = 0
    mask = torch.ByteTensor(mask)
    return mask.cuda()


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (batch_size, max_sent_len), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)
    sents_t = []
    masks = []
    for e_id in range(batch_size):
        if type(sents[0][0]) != list:
            sents_t.append([sents[e_id][i] if len(sents[e_id]) > i else pad_token for i in range(max_len)])
        else:
            sents_t.append([sents[e_id][i] if len(sents[e_id]) > i else [pad_token] for i in range(max_len)])

        masks.append([1 if len(sents[e_id]) > i else 0 for i in range(max_len)])
    # for i in range(max_len):
    #     if type(sents[0][0]) != list:
    #         sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])
    #     else:
    #         sents_t.append([sents[k][i] if len(sents[k]) > i else [pad_token] for k in xrange(batch_size)])
    #
    #     masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        if type(sents[0][0]) != list:
            return [[vocab[w] for w in s] for s in sents]
        else:
            return [[[vocab[w] for w in s] for s in v] for v in sents ]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]


def to_input_variable(sequences, vocab, cuda=False, training=True):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    word_ids = word2id(sequences, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    if type(sents_t[0][0]) != list:
        with torch.no_grad():
            sents_var = Variable(torch.LongTensor(sents_t), requires_grad=False)
        if cuda:
            sents_var = sents_var.cuda()
    else:
        # val_tok_array = np.zeros((len(sents_t), len(sents_t[0])), dtype=np.int64)
        # for i in range(len(sents_t)):
        #     for t in range(len(sents_t[0])):
        #         val_tok_array[i, t] = sents_t[i][t]
        # val_tok = torch.from_numpy(val_tok_array)
        # if cuda:
        #     val_tok = val_tok.cuda()
        sents_var = sents_t

    return sents_var


def variable_constr(x, v, cuda=False):
    return Variable(torch.cuda.x(v)) if cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in xrange(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       source: https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal(p.data)


def identity(x):
    return x


def pad_matrix(matrixs, cuda=False):
    """
    :param matrixs:
    :return: [batch_size, max_shape, max_shape], [batch_size]
    """
    shape = [m.shape[0] for m in matrixs]
    max_shape = max(shape)
    tensors = list()
    for s, m in zip(shape, matrixs):
        delta = max_shape - s
        if s > 0:
            tensors.append(torch.as_tensor(np.pad(m, [(0, delta), (0, delta)], mode='constant'), dtype=torch.float))
        else:
            tensors.append(torch.as_tensor(m, dtype=torch.float))
    tensors = torch.stack(tensors)
    if cuda:
        tensors = tensors.cuda()
    return tensors
