import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np

def process_seqs(iseqs, idates, ses_id):# make sub graph
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    max_length = 0
    for seq, date, ses in zip(iseqs, idates, ses_id):
        if len(seq) > max_length:
            max_length = len(seq)
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [ses]
    return out_seqs, out_dates, labs, ids, max_length

def process_seqs_5over_sub_graph(iseqs, idates, ses_id):# make sub graph
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    max_length = 0
    for seq, date, ses in zip(iseqs, idates, ses_id):
        length = len(seq)
        if length > max_length:
            max_length = length
        if length < 5:
            continue
        for i in range(1, length - 5):
            tar = seq[length-1]
            labs += [tar]
            out_seqs += [seq[i:length - 1]]
            out_dates += [date]
            ids += [ses]
    return out_seqs, out_dates, labs, ids, max_length

def process_seqs_5over_sub_graph_cy(iseqs, idates, ses_id):# make sub graph
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    max_length = 0
    for seq, date, ses in zip(iseqs, idates, ses_id):
        length = len(seq)
        if length > max_length:
            max_length = length
        if length < 5:
            continue
        for i in range(5, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [ses]
    return out_seqs, out_dates, labs, ids, max_length

def process_seqs_no_sub_graph(iseqs, idates, ses_id):# make sub graph
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    max_length = 0
    for seq, date, ses in zip(iseqs, idates, ses_id):
        length = len(seq)
        if length > max_length:
            max_length = length
        tar = seq[length-1]
        labs += [tar]
        out_seqs += [seq[:length - 1]]
        out_dates += [date]
        ids += [ses]
    return out_seqs, out_dates, labs, ids, max_length

def get(filepath):
    dataset = filepath

    # return format: dict(user_id : [item seq])
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        sess_clicks, sess_date, ctr, curid, curdate = {}, {}, 0, -1, None

        for data in reader:
            sessid = data['session_id']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid

            item = data['item_id'], int(data['timeframe'])
            curdate = ''
            curdate = data['eventdate']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date

    # remove if item seq == 1
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

    # item 등장 횟수 5회 미만일 경우 해당 아이템 제외, 그리고 item seq 길이 다시 측정 후 remove if item seq == 1
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    dates = np.array(list(sess_date.items()))


    index = np.arange(dates.shape[0])
    np.random.shuffle(index)
    train_size = int(len(index) * 0.8)
    train_idx = index[:train_size]
    test_idx = index[train_size:]
    
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]
    
    item_dict = {}

    tra_ids = []
    tra_dates = []
    tra_seqs = []
    
    item_ctr = 1

    for obj in train_dates:
        seq = sess_clicks[obj[0]]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        tra_ids += [obj[0]]
        tra_dates += [obj[1]]
        tra_seqs += [outseq]

    tes_ids = []
    tes_dates = []
    tes_seqs = []
    for obj in test_dates:
        seq = sess_clicks[obj[0]]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        tes_ids += [obj[0]]
        tes_dates += [obj[1]]
        tes_seqs += [outseq]
    train_data_obj = (tra_ids, tra_dates, tra_seqs)
    test_data_odj = (tes_ids, tes_dates, tes_seqs)
    tr_seqs, tr_dates, tr_labs, tr_ids, tr_max_len = process_seqs_5over_sub_graph_cy(tra_seqs, tra_dates, tra_ids)
    te_seqs, te_dates, te_labs, te_ids, ts_max_len = process_seqs_no_sub_graph(tes_seqs, tes_dates, tes_ids)
    tra = (tr_seqs, tr_labs, tr_max_len)
    tes = (te_seqs, te_labs, ts_max_len)

    return tra, tes, train_data_obj, test_data_odj

def get_k(filepath, k):
    dataset = filepath

    # return format: dict(user_id : [item seq])
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        sess_clicks, sess_date, ctr, curid, curdate = {}, {}, 0, -1, None

        for data in reader:
            sessid = data['session_id']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid

            item = data['item_id'], int(data['timeframe'])
            curdate = ''
            curdate = data['eventdate']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date

    # remove if item seq == 1
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
    
    # item 등장 횟수 5회 미만일 경우 해당 아이템 제외, 그리고 item seq 길이 다시 측정 후 remove if item seq == 1
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    dates = np.array(list(sess_date.items()))

    index = np.arange(dates.shape[0])
    np.random.shuffle(index)
    index = np.array_split(index, k)

    ret = []

    for i in range(k):
        ret.append(dates[index[i]])
    
    return ret, sess_clicks

def make_tr_ts_data_set(tr, ts, sess_clicks):
    train_dates = np.array(tr[0])
    test_dates = np.array(ts)
    tr = tr[1:]
    for tr_ in tr:
        train_dates = np.concatenate([train_dates, tr_])
    
    item_dict = {}

    tra_ids = []
    tra_dates = []
    tra_seqs = []
    
    item_ctr = 1

    for obj in train_dates:
        seq = sess_clicks[obj[0]]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        tra_ids += [obj[0]]
        tra_dates += [obj[1]]
        tra_seqs += [outseq]

    tes_ids = []
    tes_dates = []
    tes_seqs = []
    for obj in test_dates:
        seq = sess_clicks[obj[0]]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        tes_ids += [obj[0]]
        tes_dates += [obj[1]]
        tes_seqs += [outseq]
    train_data_obj = (tra_ids, tra_dates, tra_seqs)
    test_data_odj = (tes_ids, tes_dates, tes_seqs)
    tr_seqs, tr_dates, tr_labs, tr_ids, tr_max_len = process_seqs(tra_seqs, tra_dates, tra_ids)
    te_seqs, te_dates, te_labs, te_ids, ts_max_len = process_seqs_no_sub_graph(tes_seqs, tes_dates, tes_ids)
    tra = (tr_seqs, tr_labs, tr_max_len)
    tes = (te_seqs, te_labs, ts_max_len)

    return tra, tes, train_data_obj, test_data_odj

