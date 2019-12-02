# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch
from torch.autograd import Variable
from six.moves import xrange

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
