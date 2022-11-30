import time
import csv
import pickle
import operator
import datetime
import os

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


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


    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]

    for _, date in dates:
        if maxdate < date:
            maxdate = date

    # 7 days for test
    splitdate = maxdate - 86400 * 7

    print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]

    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    item_dict = {}

    # Convert training sessions to sequences and renumber items to start from 1
    def obtian_tra():
        train_ids = []
        train_seqs = []
        train_dates = []
        item_ctr = 1
        for s, date in tra_sess:
            seq = sess_clicks[s]
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
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
    #     print(item_ctr)     # 43098, 37484
        return train_ids, train_dates, train_seqs


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_tes():
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in tes_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs

    tra_ids, tra_dates, tra_seqs = obtian_tra()
    tes_ids, tes_dates, tes_seqs = obtian_tes()

    tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)

    return tra, tes