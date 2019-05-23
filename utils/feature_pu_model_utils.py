# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 13:30
# @Author  : Xiaoyu Xing
# @File    : feature_pu_model_utils.py
from utils.plain_model_utils import ModelUtils
import numpy as np


class FeaturedDetectionModelUtils(ModelUtils):
    def __init__(self, dp):
        super(FeaturedDetectionModelUtils, self).__init__()
        self.dp = dp

    def add_dict_info(self, sentences, windowSize, datasetName):
        perBigDic = set()
        locBigDic = set()
        orgBigDic = set()
        miscBigDic = set()
        with open("feature_dictionary/" + datasetName + "/personBigDic.txt", "r",encoding='utf-8') as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    perBigDic.add(line)
        with open("feature_dictionary/" + datasetName + "/locationBigDic.txt", "r",encoding='utf-8') as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    locBigDic.add(line)
        with open("feature_dictionary/" + datasetName + "/organizationBigDic.txt", "r",encoding='utf-8') as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    orgBigDic.add(line)
        if self.dp.dataset != "muc" and self.dp.dataset != "twitter":
            with open("feature_dictionary/" + datasetName + "/miscBigDic.txt", "r",encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        miscBigDic.add(line)
        for i, sentence in enumerate(sentences):
            for j, data in enumerate(sentence):
                feature = np.zeros([4, windowSize], dtype=int)
                maxLen = len(sentence)
                remainLenRight = maxLen - j - 1
                rightSize = min(remainLenRight, windowSize - 1)
                remainLenLeft = j
                leftSize = min(remainLenLeft, windowSize - 1)
                k = 0
                words = []
                words.append(sentence[j][0])

                while k < rightSize:
                    # right side
                    word = sentence[j + k + 1][0]
                    temp = words[-1]
                    word = temp + " " + word
                    words.append(word)
                    k += 1

                k = 0
                while k < leftSize:
                    # left side
                    word = sentence[j - k - 1][0]
                    temp = words[0]
                    word = word + " " + temp
                    words.insert(0, word)
                    k += 1

                for idx, word in enumerate(words):
                    count = len(word.split())
                    if word in perBigDic:
                        feature[self.dp.tag2Idx["PER"]][count - 1] = 1
                    elif word in locBigDic:
                        feature[self.dp.tag2Idx["LOC"]][count - 1] = 1
                    elif word in orgBigDic:
                        feature[self.dp.tag2Idx["ORG"]][count - 1] = 1
                feature = feature.reshape([-1]).tolist()
                sentences[i][j] = [data[0], data[1], feature, data[2], data[3]]

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
            featureList = []
            entityFlags = []
            labeledFlags = []

            for word, char, feature, ef, lf in sentence:
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
                featureList.append(feature)
                entityFlags.append(ef)
                labeledFlags.append(lf)

            dataset.append(
                [wordIndices, caseIndices, charIndices, featureList, entityFlags, labeledFlags])
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
            sentences_X.append(sentence[:4])
            sentences_Y.append(sentence[4])
            sentences_LF.append(sentence[5])
        return np.array(sentences_X), np.array(sentences_Y), np.array(sentences_LF)

    def make_PU_dataset(self, dataset):

        def _make_PU_dataset(x, y, flag):
            n_labeled = 0
            n_unlabeled = 0
            all_item = 0
            for item in flag:
                item = np.array(item)
                n_labeled += (item == 1).sum()
                item = np.array(item)
                n_unlabeled += (item == 0).sum()
                all_item += len(item)

            labeled = n_labeled
            unlabeled = n_unlabeled
            labels = np.array([0, 1])
            positive, negative = labels[1], labels[0]
            n_p = 0
            n_lp = labeled
            n_n = 0
            n_u = unlabeled
            for li in y:
                li = np.array(li)
                count = (li == positive).sum()
                n_p += count
                count2 = (li == negative).sum()
                n_n += count2

            if labeled + unlabeled == all_item:
                n_up = n_p - n_lp
            elif unlabeled == all_item:
                n_up = n_p
            else:
                raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
            prior = float(n_up) / float(n_u)
            print(prior)
            return x, y, flag, prior

        (_train_X, _train_Y, _labeledFlag), (_, _, _), (_, _, _) = dataset
        X, Y, FG, prior = _make_PU_dataset(_train_X, _train_Y, _labeledFlag)
        return list(zip(X, Y, FG)), prior

    def load_dataset(self, flag, datasetName, percent):
        fname = "data/" + datasetName + "/train." + flag + ".txt"
        trainSentences = self.dp.read_processed_file(fname, flag)
        trainSize = int(len(trainSentences) * percent)
        trainSentences = trainSentences[:trainSize]
        self.add_char_info(trainSentences)
        self.add_dict_info(trainSentences, 3, datasetName)
        train_sentences_X, train_sentences_Y, train_sentences_LF = self.padding(
            self.createMatrices(trainSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        validSentences = self.dp.read_processed_file("data/" + datasetName + "/valid.txt", flag)
        self.add_char_info(validSentences)
        self.add_dict_info(validSentences, 3, datasetName)
        valid_sentences_X, valid_sentences_Y, valid_sentences_LF = self.padding(
            self.createMatrices(validSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        testSentences = self.dp.read_processed_file("data/" + datasetName + "/test.txt", flag)
        self.add_char_info(testSentences)
        self.add_dict_info(testSentences, 3, datasetName)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(testSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        dataset = ((train_sentences_X, train_sentences_Y, train_sentences_LF),
                   (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                   (test_sentences_X, test_sentences_Y, test_sentences_LF))

        trainSet, prior = self.make_PU_dataset(dataset)
        trainX, trainY, FG = zip(*trainSet)
        trainSet = list(zip(trainX, trainY, FG))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, prior

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
                features = []
                labels = []
                flags = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                data_FG = flag[start_index:end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                for dt in data_Y:
                    dt = np.array(dt)
                    dt = (dt).astype('int32')
                    labels.append(np.eye(2)[dt])
                for dt in data_FG:
                    dt = np.array(dt)
                    dt = (dt).astype('int32')
                    flags.append(np.eye(2)[dt])

                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(features), np.asarray(
                    labels), np.asarray(
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
                features = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(features), np.asarray(
                    data_Y)
