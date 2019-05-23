# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 12:43
# @Author  : Xiaoyu Xing
# @File    : dataUtils.py

import numpy as np
import os


class DataPrepare(object):
    def __init__(self, dataset):
        self.tag2Idx = {
            "PER": 0, "LOC": 1, "ORG": 2, "MISC": 3
        }
        self.idx2tag = {
            0: "PER", 1: "LOC", 2: "ORG", 3: "MISC"
        }
        self.case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                         'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(self.case2Idx), dtype='float32')
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            self.char2Idx[c] = len(self.char2Idx)
        self.words = self.get_words(dataset)
        self.word2Idx = {}
        self.wordEmbeddings = []
        with open("data/glove.6B.100d.txt", "r", encoding='utf-8') as fw:
            for line in fw:
                line = line.strip()
                splits = line.split(" ")

                if len(self.word2Idx) == 0:
                    self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
                    vector = np.zeros(len(splits) - 1)  # Zero vector vor 'PADDING' word
                    self.wordEmbeddings.append(vector)

                    self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
                    vector = np.random.uniform(-0.25, 0.25, len(splits) - 1)
                    self.wordEmbeddings.append(vector)

                if splits[0].lower() in self.words:
                    vector = np.array([float(num) for num in splits[1:]])
                    self.wordEmbeddings.append(vector)
                    self.word2Idx[splits[0]] = len(self.word2Idx)
        self.wordEmbeddings = np.array(self.wordEmbeddings)
        self.dataset = dataset

    def read_origin_file(self, filename):
        with open(filename, "r", encoding='utf-8') as fw:
            sentences = []
            sentence = []
            for line in fw:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    splits = line.split(' ')
                    sentence.append([splits[0].strip(), splits[1].strip(), np.zeros(4)])

            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

            return sentences

    def read_processed_file(self, filename, flag):
        """
        return data [[[word, isEntity, isLabeled],[],...],...]
        :param filename:
        :return:
        """
        with open(filename, "r", encoding='utf-8') as fw:
            sentences = []
            sentence = []
            # i = 0
            for line in fw:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    # word, trueEntityLabel, dicEntityLabel
                    splits = line.split(' ')
                    if len(splits[0].strip()) > 0:
                        if splits[1].strip() != "-1":
                            sentence.append([splits[0].strip(), int(
                                splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag), int(splits[2])])
                        else:
                            sentence.append([splits[0].strip(), -1, int(splits[2])])

                    else:
                        if splits[1].strip() != "-1":
                            sentence.append([" ", int(
                                splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag), int(splits[2])])
                        else:
                            sentence.append([splits[0].strip(), -1, int(splits[2])])
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            # print(i)

            return sentences

    def writeFile(self, fileName, mode, dic, sentences):
        if mode == "TRAIN":
            with open(fileName, "w") as fw:
                for sentence in sentences:
                    for word, label, tagIdxList in sentence:
                        if np.sum(tagIdxList == True) == 1 and tagIdxList[self.tag2Idx[dic]] == 1:
                            labeled = 1
                            fw.write(word + " " + label + " " + str(labeled) + "\n")
                        else:
                            labeled = 0
                            fw.write(word + " " + label + " " + str(labeled) + "\n")
                    fw.write("\n")
        else:
            with open(fileName, "w") as fw:
                for sentence in sentences:
                    for word, label, tagIdxList in sentence:
                        labeled = 0
                        fw.write(word + " " + label + " " + str(labeled) + "\n")
                    fw.write("\n")

    def wordLevelGeneration(self, sentences):
        newSentences = []
        newLabels = []
        newPreds = []
        for sentence in sentences:
            words = []
            labels = []
            preds = []
            for i, (word, label, pred) in enumerate(sentence):
                phase = [word]
                phase_label = [label]
                phase_pred = [pred]
                if label != 'O':
                    splits = label.split("-")
                    tag = splits[0]
                    entityLabel = splits[1]
                    if tag == 'B':
                        j = i + 1
                        while j < len(sentence):
                            if sentence[j][1] != 'O':
                                tag2 = sentence[j][1].split('-')[0]
                                entityLabel2 = sentence[j][1].split('-')[1]
                                if tag2 == 'I' and entityLabel2 == entityLabel:
                                    phase = phase + [sentence[j][0]]
                                    phase_label += [sentence[j][1]]
                                    phase_pred += [sentence[j][2]]
                                    j += 1
                                    if j == len(sentence):
                                        words.append(phase)
                                        labels.append(phase_label)
                                        preds.append(phase_pred)
                                        break
                                else:
                                    words.append(phase)
                                    labels.append(phase_label)
                                    preds.append(phase_pred)
                                    break
                            else:
                                words.append(phase)
                                labels.append(phase_label)
                                preds.append(phase_pred)
                                break
                        if j - i == 1 and j == len(sentence):
                            words.append(phase)
                            labels.append(phase_label)
                            preds.append(phase_pred)
                            i += 1
                            break
                    assert len(phase) == len(phase_label) == len(phase_pred)

                else:
                    words.append(phase)
                    labels.append(phase_label)
                    preds.append(phase_pred)
            newSentences.append(words)
            newLabels.append(labels)
            newPreds.append(preds)

        newLabels_ = []
        for s in newLabels:
            temp = []
            for item in s:
                if len(item) == 1 and item[0] != "O":
                    label = item[0].split("-")[-1].strip()
                    temp.append([label])
                elif len(item) > 1:
                    temp2 = []
                    for j in item:
                        newJ = j.split("-")[-1].strip()
                        temp2.append(newJ)
                    temp.append(temp2)
                elif len(item) == 1 and item[0] == "O":
                    temp.append([item[0]])
            newLabels_.append(temp)

        return newSentences, newLabels_, newPreds

    def wordLevelGeneration2(self, sentences):
        """
        has probablity used in eval
        :param sentences:
        :return:
        """
        newSentences = []
        newLabels = []
        newPreds = []
        newProbs = []
        for sentence in sentences:
            words = []
            labels = []
            preds = []
            probs = []
            for i, (word, label, pred, prob) in enumerate(sentence):
                phase = [word]
                phase_label = [label]
                phase_pred = [pred]
                phase_prob = [prob]
                if label != 'O':
                    splits = label.split("-")
                    tag = splits[0]
                    entityLabel = splits[1]
                    if tag == 'B':
                        j = i + 1
                        while j < len(sentence):
                            if sentence[j][1] != 'O':
                                tag2 = sentence[j][1].split('-')[0]
                                entityLabel2 = sentence[j][1].split('-')[1]
                                if tag2 == 'I' and entityLabel2 == entityLabel:
                                    phase = phase + [sentence[j][0]]
                                    phase_label += [sentence[j][1]]
                                    phase_pred += [sentence[j][2]]
                                    phase_prob += [sentence[j][3]]
                                    j += 1
                                    if j == len(sentence):
                                        words.append(phase)
                                        labels.append(phase_label)
                                        preds.append(phase_pred)
                                        probs.append(phase_prob)
                                        break
                                else:
                                    words.append(phase)
                                    labels.append(phase_label)
                                    preds.append(phase_pred)
                                    probs.append(phase_prob)
                                    break
                            else:
                                words.append(phase)
                                labels.append(phase_label)
                                preds.append(phase_pred)
                                probs.append(phase_prob)
                                break
                        if j - i == 1 and j == len(sentence):
                            words.append(phase)
                            labels.append(phase_label)
                            preds.append(phase_pred)
                            probs.append(phase_prob)
                            i += 1
                            break
                    assert len(phase) == len(phase_label) == len(phase_pred) == len(phase_prob)

                else:
                    words.append(phase)
                    labels.append(phase_label)
                    preds.append(phase_pred)
                    probs.append(phase_prob)
            newSentences.append(words)
            newLabels.append(labels)
            newPreds.append(preds)
            newProbs.append(probs)

        newLabels_ = []
        for s in newLabels:
            temp = []
            for item in s:
                if len(item) == 1 and item[0] != "O":
                    label = item[0].split("-")[-1].strip()
                    temp.append([label])
                elif len(item) > 1:
                    temp2 = []
                    for j in item:
                        newJ = j.split("-")[-1].strip()
                        temp2.append(newJ)
                    temp.append(temp2)
                elif len(item) == 1 and item[0] == "O":
                    temp.append([item[0]])
            newLabels_.append(temp)

        return newSentences, newLabels_, newPreds, newProbs

    def compute_precision_recall_f1(self, labels, preds, flag, pflag):
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
        return p, r, f1



    def get_words(self, dataset):
        words = {}
        if dataset == "conll2003" or dataset == 'conll2002':
            trainSentences = self.read_origin_file("data/" + dataset + "/train.txt")
            validSentences = self.read_origin_file("data/" + dataset + "/valid.txt")
            testSentences = self.read_origin_file("data/" + dataset + "/test.txt")
            for sentences in [trainSentences, validSentences, testSentences]:
                for sentence in sentences:
                    for token, label, flag in sentence:
                        words[token.lower()] = True
            return words
        elif dataset == "muc" or dataset == "wikigold" or dataset == "twitter":
            trainSentences = self.read_origin_file("data/" + dataset + "/train.txt")
            testSentences = self.read_origin_file("data/" + dataset + "/test.txt")
            for sentences in [trainSentences, testSentences]:
                for sentence in sentences:
                    for token, label, flag in sentence:
                        words[token.lower()] = True
            return words
        else:
            raise ValueError("dataset name is wrong")

    def read_unlabeled_data(self, filename):
        with open(filename, "r") as fw:
            sentences = []
            sentence = []
            for line in fw:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    splits = line.split(' ')
                    sentence.append([splits[0].strip(), -1, np.zeros(4)])

            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

            return sentences
