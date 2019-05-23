# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 13:43
# @Author  : Xiaoyu Xing
# @File    : modelUtils.py
import numpy as np


class ModelUtils(object):
    def __init__(self):
        pass

    def get_casing(self, word, caseLookup):
        casing = 'other'
        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():
            casing = 'numeric'
        elif digitFraction > .5:
            casing = 'mainly_numeric'
        elif word.islower():
            casing = 'allLower'
        elif word.isupper():
            casing = 'allUpper'
        elif word[0].isupper():
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return caseLookup[casing]

    def add_char_info(self, sentences):
        for i, sentence in enumerate(sentences):
            for j, data in enumerate(sentence):
                chars = [c for c in data[0]]
                sentences[i][j] = [data[0], chars, data[1], data[2]]

    def createMatrices(self, sentences, word2Idx, case2Idx, char2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']
        paddingIdx = word2Idx['PADDING_TOKEN']

        dataset = []

        wordCount = 0
        unknownWordCount = 0

        for sentence in sentences:
            wordIndices = []
            caseIndices = []
            charIndices = []
            entityFlags = []
            labeledFlags = []

            for word, char, ef, lf in sentence:
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                charIdx = []
                for x in char:
                    if x in char2Idx:
                        charIdx.append(char2Idx[x])
                    else:
                        charIdx.append(char2Idx["UNKNOWN"])

                wordIndices.append(wordIdx)
                caseIndices.append(self.get_casing(word, case2Idx))
                charIndices.append(charIdx)
                entityFlags.append(ef)
                labeledFlags.append(lf)

            dataset.append(
                [wordIndices, caseIndices, charIndices, entityFlags, labeledFlags])
        return dataset

    def padding(self, sentences):
        maxlen = 52
        for i, sentence in enumerate(sentences):
            mask = np.zeros([len(sentences[i][2]), maxlen])
            for j, chars in enumerate(sentences[i][2]):
                for k, c in enumerate(chars):
                    if k < maxlen:
                        mask[j][k] = c
            sentences[i][2] = mask.tolist()

        sentences_X = []
        sentences_Y = []
        sentences_LF = []

        for i, sentence in enumerate(sentences):
            sentences_X.append(sentence[:3])
            sentences_Y.append(sentence[3])
            sentences_LF.append(sentence[4])
        return np.array(sentences_X), np.array(sentences_Y), np.array(sentences_LF)

    def iterateSet(self, trainset, batchSize, mode, shuffle=True):
        if mode == "TRAIN":
            data_size = len(trainset)
            X, Y, FG = zip(*trainset)
            X = np.array(X)
            Y = np.array(Y)
            FG = np.array(FG)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = np.array(X)[shuffle_indices]
                y = np.array(Y)[shuffle_indices]
                flag = np.array(FG)[shuffle_indices]
            else:
                x = X
                y = Y
                flag = FG

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batchSize
                end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                labels = []
                flags = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                data_FG = flag[start_index:end_index]
                for dt in data_X:
                    t, c, ch = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                for dt in data_Y:
                    dt = np.array(dt)
                    dt = (dt).astype('int32')
                    labels.append(np.eye(2)[dt])
                for dt in data_FG:
                    dt = np.array(dt)
                    dt = (dt).astype('int32')
                    flags.append(np.eye(2)[dt])

                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(labels), np.asarray(
                    flags)
        else:
            data_size = len(trainset)
            X, Y, _ = zip(*trainset)
            X = np.array(X)
            Y = np.array(Y)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = np.array(X)[shuffle_indices]
                y = np.array(Y)[shuffle_indices]
            else:
                x = X
                y = Y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batchSize
                end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                for dt in data_X:
                    t, c, ch = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(data_Y)
