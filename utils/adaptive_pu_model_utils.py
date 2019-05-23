# -*- coding: utf-8 -*-
# @Time    : 2018/8/8 17:21
# @Author  : Xiaoyu Xing
# @File    : adaptive_pu_model_utils.py

from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
from collections import defaultdict
import numpy as np


class AdaptivePUUtils(FeaturedDetectionModelUtils):
    def __init__(self, dp):
        super(AdaptivePUUtils, self).__init__(dp)
        self.dp = dp

    def adaptive_word_level_generation(self, trainSet, sentences, value=0.8):
        trainX, trainY, FG = zip(*trainSet)

        newSentences = []
        newLabels = []
        newPreds = []
        newRes = []
        for sentence in sentences:
            words = []
            labels = []
            preds = []
            ress = []
            for i, (word, label, pred, res) in enumerate(sentence):
                phase = [word]
                phase_label = [label]
                phase_pred = [pred]
                phase_res = [res]
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
                                    phase_res += [sentence[j][3]]
                                    j += 1
                                    if j == len(sentence):
                                        words.append(phase)
                                        labels.append(phase_label)
                                        preds.append(phase_pred)
                                        ress.append(phase_res)
                                        break
                                else:
                                    words.append(phase)
                                    labels.append(phase_label)
                                    preds.append(phase_pred)
                                    ress.append(phase_res)
                                    break
                            else:
                                words.append(phase)
                                labels.append(phase_label)
                                preds.append(phase_pred)
                                ress.append(phase_res)
                                break
                        if j - i == 1 and j == len(sentence):
                            words.append(phase)
                            labels.append(phase_label)
                            preds.append(phase_pred)
                            ress.append(phase_res)
                            i += 1
                            break
                    assert len(phase) == len(phase_label) == len(phase_pred) == len(phase_res)

                else:
                    words.append(phase)
                    labels.append(phase_label)
                    preds.append(phase_pred)
                    ress.append(phase_res)
            newSentences.append(words)
            newLabels.append(labels)
            newPreds.append(preds)
            newRes.append(ress)

        for i, res in enumerate(newRes):
            # print(res)
            for j, item in enumerate(res[0]):
                if item[1] - item[0] >= value and FG[i][j] == 0:
                    FG[i][j] = 1

        trainSet = list(zip(trainX, trainY, FG))

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

        return newSentences, newLabels_, newPreds, trainSet

    def get_true_occur(self, wordList, trainX):
        trueOccur = defaultdict(int)
        for i, sentence in enumerate(trainX):
            j = 0
            while j < len(sentence):
                if sentence[j] == wordList[0]:
                    k = 1
                    allEql = True
                    while k < len(wordList) and j + k < len(sentence):
                        if sentence[j + k] != wordList[k]:
                            allEql = False
                            break
                        else:
                            k += 1
                    if allEql:
                        words = " ".join([w for w in wordList])
                        trueOccur[words] += 1
                j += 1
        return trueOccur

    def revise_dictionary(self, trainSet, dictionaryFile, newDicFile):
        trainX, predY = zip(*trainSet)
        allphaseLists = defaultdict(int)
        labeledphaseLists = defaultdict(int)
        dictionary = set()
        with open(dictionaryFile, "r") as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    dictionary.add(line)

        size1 = len(dictionary)

        for i, sentence in enumerate(trainX):
            j = 0
            phase = []
            phase_pred = []
            while j < len(sentence):
                if predY[i][j] == 1:
                    p = [trainX[i][j]]
                    p_pred = [predY[i][j]]
                    k = j + 1
                    while k < len(sentence):
                        if predY[i][k] == 1:
                            p += [trainX[i][k]]
                            p_pred += [predY[i][k]]
                        else:
                            j = k
                            break
                        k += 1

                    phase.append(p)
                    phase_pred.append(p_pred)
                else:
                    phase.append([trainX[i][j]])
                    phase_pred.append([predY[i][j]])
                j += 1
            for m, p in enumerate(phase):
                p_entity = " ".join([w for w in p])
                # allphaseLists[p_entity] += 1
                pred_p = phase_pred[m]
                if np.array(pred_p).all() == 1:
                    labeledphaseLists[p_entity] += 1

        for words in labeledphaseLists:
            labeledoccr = labeledphaseLists[words]
            wordslist = words.split(" ")
            trueoccur = self.get_true_occur(wordslist, trainX)
            # print(trueoccur)

            if trueoccur[words] == labeledoccr and labeledoccr >= 3:
                dictionary.add(words)

        size2 = len(dictionary)
        print("add " + str(size2 - size1) + " words in dictionary")

        with open(newDicFile, "w") as fw:
            for d in dictionary:
                fw.write(d + "\n")

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
            return x, y, flag, n_lp

        (_train_X, _train_Y, _labeledFlag), (_, _, _), (_, _, _) = dataset
        X, Y, FG, n_lp = _make_PU_dataset(_train_X, _train_Y, _labeledFlag)
        return list(zip(X, Y, FG)), n_lp

    def load_new_dataset(self, flag, datasetName, iter, p):
        fname = "data/" + datasetName + "/train." + flag + str(iter) + ".txt"
        trainSentences = self.dp.read_processed_file(fname, flag)
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

        trainSet, n_lp = self.make_PU_dataset(dataset)

        n = 0
        for i, sentence in enumerate(train_sentences_X):
            n += len(sentence[0])

        prior = float(n * p - n_lp) / float(n - n_lp)

        trainX, trainY, FG = zip(*trainSet)
        trainSet = list(zip(trainX, trainY, FG))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, prior

    def read_unlabeledset(self, datasetName):
        fname = "data/" + datasetName + "/unlabeled/train.txt"
        sentences = self.dp.read_origin_file(fname)
        return sentences

    def createMatrices2(self, sentences, word2Idx, case2Idx, char2Idx):
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
            # print(sentence)

            for word, char, feature, _, _ in sentence:
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
                entityFlags.append(-1)
                labeledFlags.append(0)

            dataset.append(
                [wordIndices, caseIndices, charIndices, featureList, entityFlags, labeledFlags])
        return dataset

    def load_unlabeledset(self, sentences, datasetName):
        self.add_char_info(sentences)
        self.add_dict_info(sentences, 3, datasetName)

        train_sentences_X_unlabeled, train_sentences_Y_unlabeled, train_sentences_LF_unlabeled = self.padding(
            self.createMatrices2(sentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        unlabeledSet = list(zip(train_sentences_X_unlabeled, train_sentences_Y_unlabeled, train_sentences_LF_unlabeled))
        return unlabeledSet
