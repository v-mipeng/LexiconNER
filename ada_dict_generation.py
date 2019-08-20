# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 17:35
# @Author  : Xiaoyu Xing
# @File    : ada_dict_generation.py
from utils.data_utils import DataPrepare
from utils.adaptive_pu_model_utils import AdaptivePUUtils
from utils.dict_utils import DictUtils
import torch
import argparse
from adaptive_pu_model import Trainer, AdaPULSTMCNN2
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet
import numpy as np
import os


def new_dict_generation(mutils, dp, dutils, flag, word_predict, dataset, iteration, unlabeled=0):
    if flag == "PER":
        name = "person.txt"
    elif flag == "LOC":
        name = "location.txt"
    elif flag == "ORG":
        name = "organization.txt"
    elif flag == "MISC":
        name = "misc.txt"
    else:
        raise ValueError("wrong entity name")

    # print(unlabeled)

    if unlabeled:
        fname = "data/" + dataset + "/unlabeled/train.txt"
        if os.path.isdir("dictionary/" + dataset + "/unlabeled") == False:
            os.makedirs("dictionary/" + dataset + "/unlabeled")
        newDicFile = "dictionary/" + dataset + "/unlabeled/" + str(iteration) + "_" + name
    else:
        fname = "data/" + dataset + "/train.txt"
        newDicFile = "dictionary/" + dataset + "/" + str(iteration) + "_" + name
    mutils.revise_dictionary(word_predict,
                             "dictionary/" + dataset + "/" + name,
                             newDicFile)

    # newDicFile = "dictionary/" + dataset + "/unlabeled/" + "test_" + str(iteration) + "_" + name

    oriSentences = dp.read_origin_file(fname)

    oriSentences, _, _ = dutils.lookup_in_Dic(newDicFile, oriSentences, flag,
                                              5)

    if unlabeled:
        if os.path.isdir("data/" + dataset + "/unlabeled") == False:
            os.makedirs("data/" + dataset + "/unlabeled")

        dp.writeFile("data/" + dataset + "/unlabeled/train." + flag + str(iteration) + ".txt", "TRAIN", flag,
                     oriSentences)

    else:
        dp.writeFile("data/" + dataset + "/train." + flag + str(iteration) + ".txt", "TRAIN", flag, oriSentences)


if __name__ == "__main__":
    torch.manual_seed(10)

    parser = argparse.ArgumentParser(description="PU NER")
    # data

    parser.add_argument('--beta', type=float, default=0.0,help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0,help='gamma of pu learning (default 1.0)')
    parser.add_argument('--drop_out', type=float, default=0.5,help = 'dropout rate')
    parser.add_argument('--m', type=float, default=0.5,help='class balance rate')
    parser.add_argument('--flag', default="PER", help='entity type (PER/LOC/ORG/MISC)')
    parser.add_argument('--dataset', default="conll2003",help='name of the dataset')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--batch_size', type=int, default=300,help='batch size for training and testing')
    parser.add_argument('--iter', type=int, default=1,help='iteration time')
    parser.add_argument('--unlabeled', type=int, default=0,help='use unlabeled data or not')
    parser.add_argument('--pert', type=float, default=1.0,help='percentage of data use for training')

    parser.add_argument('--model', default="",help='saved model name')  # finetune

    args = parser.parse_args()

    dp = DataPrepare(args.dataset)
    dutils = DictUtils()

    mutils = AdaptivePUUtils(dp)
    trainSet, validSet, testSet, prior = mutils.load_dataset(args.flag, args.dataset, args.pert)
    charcnn = CharCNN(dp.char2Idx)
    wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
    casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
    featurenet = FeatureNet()
    pulstmcnn = AdaPULSTMCNN2(dp, charcnn, wordnet, casenet, featurenet, 150, 200, 1, args.drop_out)

    if torch.cuda.is_available:
        charcnn.cuda()
        wordnet.cuda()
        casenet.cuda()
        featurenet.cuda()
        pulstmcnn.cuda()

    trainer = Trainer(pulstmcnn, prior, args.beta, args.gamma, args.lr, args.m)
    pulstmcnn.load_state_dict(torch.load(args.model))


    newSet = trainSet
    if args.unlabeled:
        unlabeledSet = mutils.load_unlabeledset(mutils.read_unlabeledset(args.dataset), args.dataset)
        newSet = unlabeledSet

    trainSize = len(trainSet)
    validSize = len(validSet)
    testSize = len(testSet)
    print(("train set size: {}, valid set size: {}, test set size: {}").format(trainSize, validSize, testSize))

    if args.unlabeled:
        train_sentences = dp.read_origin_file("data/" + args.dataset + "/unlabeled/train.txt")
    else:
        train_sentences = dp.read_origin_file("data/" + args.dataset + "/train.txt")
    train_words = []
    train_efs = []
    for s in train_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        train_words.append(temp)
        train_efs.append(temp2)

    test_sentences = dp.read_origin_file("data/" + args.dataset + "/test.txt")
    test_words = []
    test_efs = []
    for s in test_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        test_words.append(temp)
        test_efs.append(temp2)

    # origin result
    pred_test = []
    corr_test = []
    for step, (
            x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
            y_test_batch) in enumerate(
        mutils.iterateSet(testSet, batchSize=100, mode="TEST", shuffle=False)):
        testBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
        correcLabels = []
        for x in y_test_batch:
            for xi in x:
                correcLabels.append(xi)
        lengths = [len(x) for x in x_word_test_batch]
        predLabels,_ = trainer.test(testBatch, lengths)
        correcLabels = np.array(correcLabels)
        assert len(predLabels) == len(correcLabels)

        start = 0
        for i, l in enumerate(lengths):
            end = start + l
            p = predLabels[start:end]
            c = correcLabels[start:end]
            pred_test.append(p)
            corr_test.append(c)
            start = end

    newSentencesTest = []
    for i, s in enumerate(test_words):
        sent = []
        assert len(s) == len(test_efs[i]) == len(pred_test[i])
        for j, item in enumerate(s):
            sent.append([item, test_efs[i][j], pred_test[i][j]])
        newSentencesTest.append(sent)

    newSentencesValid_, newLabelsValid, newPredsValid = dp.wordLevelGeneration(newSentencesTest)
    p_valid, r_valid, f1_valid = dp.compute_precision_recall_f1(newLabelsValid, newPredsValid, args.flag,
                                                                1)
    print("Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))

    # revise dictionary
    pred_train = []
    for step, (x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch,
               y_train_batch) in enumerate(
        mutils.iterateSet(newSet, batchSize=100, mode="TEST", shuffle=False)):
        trainBatch = [x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch]

        lengths = [len(x) for x in x_word_train_batch]
        predLabels,_ = trainer.test(trainBatch, lengths)

        start = 0
        for i, l in enumerate(lengths):
            end = start + l
            p = predLabels[start:end]
            pred_train.append(p)
            start = end

    newSentences = []
    for i, s in enumerate(train_words):
        sent = []
        assert len(s) == len(train_efs[i]) == len(pred_train[i])
        for j, item in enumerate(s):
            sent.append([item, train_efs[i][j], pred_train[i][j]])
        newSentences.append(sent)

    word_predict = list(zip(train_words, pred_train))

    new_dict_generation(mutils, dp, dutils, args.flag, word_predict, args.dataset, args.iter, args.unlabeled)
