# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 12:49
# @Author  : Xiaoyu Xing
# @File    : dict_match.py

import utils.data_utils, utils.dict_utils
import numpy as np


def compute_precision_recall_f1(labels, preds, flag, pflag):
    """
    Word level
    :param labels:
    :param preds:
    :param flag:
    :param pflag:
    :return:
    """
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels)):
        sent_label = labels[i]
        sent_pred = preds[i]
        for j in range(len(sent_label)):
            item1 = np.array(sent_pred[j])
            item2 = np.array(sent_label[j])

            if (item1 == pflag).all() == True:
                pp += 1
            if (item2 == flag).all() == True:
                np_ += 1
                if (item1 == pflag).all() == True:
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
    return p, r, f1, tp, np_, pp


def compute_precision_recall_f1_2(labels, preds, flag, pflag):
    """
    character level
    :param labels:
    :param preds:
    :param flag:
    :param pflag:
    :return:
    """
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels)):
        item1 = np.array(preds[i])
        item2 = np.array(labels[i])

        if item1 == pflag:
            pp += 1
        if item2 == flag:
            np_ += 1
            if item1 == pflag:
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
    return p, r, f1, tp, np_, pp


def getLabelsAndPreds(sentences):
    labels = []
    preds = []
    for sent in sentences:
        for word, label, pred in sent:
            if len(label.split('-')) > 1:
                label = label.split('-')[-1]
            else:
                label = label
            labels.append(label)
            preds.append(pred)
    return labels, preds


def dict_match_word(dp, dutils, fileName, dictName, flag, mode, dataset):
    sentences = dp.read_origin_file(fileName)
    length = [len(i) for i in sentences]
    maxLen = 10
    sentences, num, count = dutils.lookup_in_Dic(dictName, sentences, flag, maxLen)
    print(count)
    if mode == "TRAIN":
        dp.writeFile("data/" + dataset + "/train." + flag + ".txt", mode, flag, sentences)
    ss = []
    for sentence in sentences:
        s = []
        for i, (word, label, pred) in enumerate(sentence):
            pred = np.array(pred)
            if pred[dutils.tag2Idx[flag]] == 1:
                s.append([word, label, 1])
            else:
                s.append([word, label, 0])
        ss.append(s)
    sentences = ss
    labels, preds = getLabelsAndPreds(sentences)
    p, r, f1, tp, np_, pp = compute_precision_recall_f1_2(labels, preds, flag, 1)
    return p, r, f1, tp, np_, pp


def dict_match_result(dp, dutils, fileName, dictName, flag, mode, dataset, percent=1.0):
    sentences = dp.read_origin_file(fileName)
    size = int(len(sentences) * percent)
    sentences = sentences[:size]
    length = [len(i) for i in sentences]
    maxLen = 10
    sentences, num, _ = dutils.lookup_in_Dic(dictName, sentences, flag, maxLen)
    print(num)
    if mode == "TRAIN":
        dp.writeFile("data/" + dataset + "/valid." + flag + ".txt", mode, flag, sentences)
    ss = []
    for sentence in sentences:
        s = []
        for i, (word, label, pred) in enumerate(sentence):
            pred = np.array(pred)
            if pred[dutils.tag2Idx[flag]] == 1:
                s.append([word, label, 1])
            else:
                s.append([word, label, 0])
        ss.append(s)
    sentences = ss
    newSentences, newLabels, newPreds = dp.wordLevelGeneration(sentences)
    p, r, f1, tp, np_, pp = compute_precision_recall_f1(newLabels, newPreds, flag, 1)
    return p, r, f1, tp, np_, pp


def count_entity(dataset, type):
    filename = "dictionary/" + dataset + "/" + type + ".txt"
    s = set()
    with open(filename, "r") as fw:
        for line in fw:
            line = line.strip()
            s.add(line)
    print(type, len(s))




if __name__ == "__main__":
    dp = utils.data_utils.DataPrepare("conll2003")
    dutils = utils.dict_utils.DictUtils()
    # num = 3

    p1, r1, f11, tp, np_1, pp = dict_match_word(dp, dutils,
                                                  "data/conll2003/valid.txt",
                                                  "dictionary/conll2003/person.txt",
                                                  "PER",
                                                  "Train", "conll2003")
    print("%.4f" % p1, "%.4f" % r1, "%.4f" % f11, tp, np_1, pp)

    p2, r2, f12, tp, np_2, pp = dict_match_word(dp, dutils, "data/conll2003/test.txt",
                                                  "dictionary/conll2003/location.txt",
                                                  "LOC",
                                                  "Train", "conll2003")
    print("%.4f" % p2, "%.4f" % r2, "%.4f" % f12, tp, np_2, pp)
    #
    p3, r3, f13, tp, np_3, pp = dict_match_word(dp, dutils, "data/conll2003/test.txt",
                                                  "dictionary/conll2003/organization.txt",
                                                  "ORG", "Train", "conll2003")
    print("%.4f" % p3, "%.4f" % r3, "%.4f" % f13, tp, np_3, pp)

    p4, r4, f14, tp, np_4, pp = dict_match_word(dp, dutils, "data/conll2003/test.txt",
                                                  "dictionary/conll2003/misc.txt",
                                                  "MISC", "Train",
                                                  "conll2003")
    print("%.4f" % p4, "%.4f" % r4, "%.4f" % f14, tp, np_4, pp)

