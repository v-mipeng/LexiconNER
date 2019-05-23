# -*- coding: utf-8 -*-
# @Time    : 2018/10/13 12:21
# @Author  : Xiaoyu Xing
# @File    : final_evl.py

import numpy as np
import argparse
from utils.data_utils import DataPrepare


def get_final_result(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    column_val = ["PER", 'LOC', 'ORG', "MISC", 'O']

    result = []
    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])

            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        res = []
        for j in range(len(merge)):
            arg_max = np.argmax(merge[j])
            max = np.max(merge[j])

            if max <= 0.5:
                res.append(column_val[-1])
            else:
                res.append(column_val[arg_max])

        result.append(res)

    return result


def get_output(filename):
    with open(filename, "r",encoding='utf-8') as fw:
        ress = []
        res = []
        for line in fw:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                if len(res) > 0:
                    ress.append(res)
                    res = []
                continue
            else:
                splits = line.strip().split(' ')
                res.append(float(splits[-1].strip()))

        if len(res) > 0:
            ress.append(res)
            res = []

        return ress


def prf1(labels, preds):
    def compute_precision_recall_f1(labels, preds):
        tp = 0
        np_ = 0
        pp = 0
        for i in range(len(labels)):
            sent_label = labels[i]
            sent_pred = preds[i]
            for j in range(len(sent_label)):
                item1 = np.array(sent_pred[j])
                item2 = np.array(sent_label[j])

                if (item1 == "PER").all() == True or (item1 == "LOC").all() == True or (
                            item1 == "ORG").all() == True :
                    pp += 1

                if (item2 == "PER").all() == True or (item2 == "LOC").all() == True or (
                            item2 == "ORG").all() == True :
                    np_ += 1
                    et_t = item2[0]

                    if (item1 == et_t).all() == True:
                        tp += 1
        if pp == 0:
            p = 0
        else:
            p = float(tp) / float(pp)
        if np_ == 0:
            r = 0
        else:
            r = float(tp) / float(np_)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = float(2 * p * r) / float((p + r))
        return p, r, f1

    p, r, f1 = compute_precision_recall_f1(labels, preds)

    return p, r, f1


def get_conflict(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    conflict_num = 0
    word_num = 0

    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])

            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        temp = 0
        for j in range(len(merge)):
            word_num += 1

            if np.sum(merge[j] > 0.5) >= 2:
                temp += 1

        if temp >= 2:
            conflict_num += 1

    return conflict_num, word_num, float(conflict_num) / float(word_num)


def get_match_final_result(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    column_val = ["PER", 'LOC', 'ORG', "MISC", 'O']

    result = []
    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])

            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        res = []
        for j in range(len(merge)):
            count_one = 0
            for m in merge[j]:
                if m == 1:
                    count_one += 1
            if count_one==1:
                arg_max = np.argmax(merge[j])
                res.append(column_val[arg_max])
            else:
                res.append(column_val[-1])

        result.append(res)

    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU NER EVL")
    parser.add_argument('--dataset', default="conll2003")
    parser.add_argument('--type', default="bnpu")
    args = parser.parse_args()

    filenames = [
        'result/'+args.type+'_feature_pu_' + args.dataset + '_PER_0.txt',
        'result/'+args.type+'_feature_pu_' + args.dataset + '_LOC_0.txt',
        'result/'+args.type+'_feature_pu_' + args.dataset + '_ORG_0.txt',
        'result/'+args.type+'_feature_pu_' + args.dataset + '_MISC_0.txt'
    ]


    origin_file = "data/" + args.dataset + "/test.txt"
    dp = DataPrepare(args.dataset)

    test_sentences = dp.read_origin_file(origin_file)
    test_words = []
    test_efs = []
    lens = []
    for s in test_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        test_words.append(temp)
        test_efs.append(temp2)
        lens.append(len(s))

    per_result = get_output(filenames[0])
    loc_result = get_output(filenames[1])
    org_result = get_output(filenames[2])
    # misc_result = get_output(filenames[3])

    # get result
    final_res = get_final_result(per_result, loc_result, org_result)

    newSentencesTest = []
    for i, s in enumerate(test_words):
        sent = []
        for j, item in enumerate(s):
            sent.append([item, test_efs[i][j], final_res[i][j]])
        newSentencesTest.append(sent)

    newSentences, newLabels, newPreds = dp.wordLevelGeneration(newSentencesTest)

    p, r, f1 = prf1(newLabels, newPreds)
    print(p, r, f1)

    # get conflict
    c, w, p = get_conflict(per_result, loc_result, org_result)
    print(c, w, p)