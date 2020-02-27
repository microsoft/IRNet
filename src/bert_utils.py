# -*- coding: utf-8 -*-
"""
# @Time    : 2019/1/26 下午4:31
# @Author  : zhanzecheng
# @File    : bert_utils.py
# @Software: PyCharm
"""
import torch
from torch import nn
import numpy as np
device = torch.device("cuda")

def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)

    n_hds = []
    for i, hds11 in enumerate(nlu1_tok):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        n_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [0] * len(sub_tok)
        # if i < len(nlu1_tok)-1:

        # elif i == len(nlu1_tok)-1:
        #     tokens.append("[SEP]")
        #     segment_ids.append(1)
        # else:
        #     raise EnvironmentError

    tokens.append("[SEP]")
    segment_ids.append(0)

    # segment_ids.append(0)
    # for token in nlu1_tok:
    #     tokens.append(token)
    #     segment_ids.append(0)
    # i_ed_nlu = len(tokens)
    # tokens.append("[SEP]")
    # segment_ids.append(0)

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    return tokens, segment_ids, n_hds, i_hds

def get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1):

    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, n_hs, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # tokens: BERT intput tokens
    # i_nlu: start and end indices of question in tokens
    # i_hds: start and end indices of headers


    # get the wemb
    wemb_n = get_wemb_avg_list(i_nlu, n_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)
    # quit()
    # wemb_n = get_wemb_h(i_nlu, l_n, n_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
    #                     num_out_layers_n)
    # wemb_h = get_wemb_avg_list(i_hds, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
    #                     num_out_layers_h)
    # print(hds)
    # print(wemb_h.shape)
    # quit()
    #
    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)


    return wemb_n, wemb_h, l_n, n_hs, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx

def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.
    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length
    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.
    """

    l_n = []

    n_hs = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        n_hs.append(len(nlu_t1))
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        # tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        # t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        # nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        # for (i, token) in enumerate(nlu_t1):
        #     t_to_tt_idx1.append(
        #         len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
        #     sub_tokens = tokenizer.tokenize(token)
        #     for sub_token in sub_tokens:
        #         tt_to_t_idx1.append(i)
        #         nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        # nlu_tt.append(nlu_tt1)
        # tt_to_t_idx.append(tt_to_t_idx1)
        # t_to_tt_idx.append(t_to_tt_idx1)

        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)

        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, n_hds, i_hds1 = generate_inputs(tokenizer, nlu_t1, hds1)

        # l_n.append(len(nlu_tt1))

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        # print(i_nlu1)
        # print(i_hds1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.

        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)


        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(n_hds)
        i_hds.append(i_hds1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)
    l_n = gen_l_hpu(i_nlu)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, n_hs, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx

def get_wemb_avg_list(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b, i_hds1 in enumerate(i_nlu):
        for b1, i_hds11 in enumerate(i_hds1):
            for i_noln in range(num_out_layers_n):
                i_layer = num_hidden_layers - 1 - i_noln
                st = i_noln * hS
                ed = (i_noln + 1) * hS
                wemb_n[b, b1, st:ed] = torch.mean(all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1], :], dim=0)

    return wemb_n

def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n
    #


def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]


    return wemb_h


def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu


def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = np.zeros(len(perm), dtype=np.int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv

def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape


    # sort before packking
    l = np.array(l)
    perm_idx = np.argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float() # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]



    if return_hidden:
        # hout.shape = [number_of_directoin * num_of_layer, seq_len(=batch size), dim * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode( lstm,
                                   wemb_hpu,
                                   l_hpu,
                                   return_hidden=True,
                                   hc0=None,
                                   last_only=True )

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs