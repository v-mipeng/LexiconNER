# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 16:48
# @Author  : Xiaoyu Xing
# @File    : feature_pu_model_evl.py

from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
from utils.data_utils import DataPrepare
import torch
import argparse
from feature_pu_model import PULSTMCNN, Trainer
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet
import numpy as np

torch.manual_seed(1013)
parser = argparse.ArgumentParser(description="PU NER")
parser.add_argument('--model', default="")
parser.add_argument('--output', default=0)
parser.add_argument('--set', type=int, default=0)
parser.add_argument('--flag', default="PER")
parser.add_argument('--lr_rate', type=int, default=1e-4)
parser.add_argument('--dataset', default="conll2003")
parser.add_argument('--type', default="bnpu")
args = parser.parse_args()

dp = DataPrepare(args.dataset)
mutils = FeaturedDetectionModelUtils(dp)

trainSet, validSet, testSet, prior = mutils.load_dataset(args.flag, args.dataset, 1.0)
setIter = [testSet, validSet]
detectionSet = setIter[args.set]
fileNameIter = ["data/" + args.dataset + "/test.txt", "data/" + args.dataset + "/valid.txt"]
fileName = fileNameIter[args.set]

charcnn = CharCNN(dp.char2Idx)
wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
featurenet = FeatureNet()
pulstmcnn = PULSTMCNN(dp, charcnn, wordnet, casenet, featurenet, 150, 200, 1, 0.5)

if torch.cuda.is_available:
    charcnn.cuda()
    wordnet.cuda()
    casenet.cuda()
    featurenet.cuda()
    pulstmcnn.cuda()

pred_test = []
corr_test = []
prob_test = []

trainer = Trainer(pulstmcnn, prior, 0, 1, 1e-4, 4)

pulstmcnn.load_state_dict(torch.load(args.model))

for step, (x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_batch, y_test_batch) in enumerate(
        mutils.iterateSet(detectionSet, batchSize=100, mode="TEST", shuffle=False)):
    testBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_batch]

    correcLabels = []
    for x in y_test_batch:
        for xi in x:
            correcLabels.append(xi)
    lengths = [len(x) for x in x_word_test_batch]
    predLabels, probLabels = trainer.test(testBatch, lengths)
    correcLabels = np.array(correcLabels)
    assert len(predLabels) == len(correcLabels) == len(probLabels)

    start = 0
    for i, l in enumerate(lengths):
        end = start + l
        p = predLabels[start:end]
        c = correcLabels[start:end]
        r = probLabels[start:end]
        pred_test.append(p)
        corr_test.append(c)
        prob_test.append(r)
        start = end

assert len(pred_test) == len(corr_test) == len(prob_test)

test_sentences = dp.read_origin_file(fileName)
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

newSentencesTest = []
for i, s in enumerate(test_words):
    sent = []
    for j, item in enumerate(s):
        sent.append([item, test_efs[i][j], pred_test[i][j], prob_test[i][j]])
    newSentencesTest.append(sent)

newSentencesTest_, newLabelsTest, newPredsTest, newProbTest = dp.wordLevelGeneration2(newSentencesTest)
p, r, f1 = dp.compute_precision_recall_f1(newLabelsTest, newPredsTest, args.flag, 1)
print("Precision: {}, Recall: {}, F1: {}".format(p, r, f1))

if args.output:
    outputFile = "result/" + args.type + "_feature_pu_" + args.dataset + "_" + args.flag + "_" + str(args.set) + ".txt"
    with open(outputFile, "w") as fw:
        for i, sent in enumerate(test_words):
            preds = pred_test[i]
            probs = prob_test[i]
            corrs = test_efs[i]
            for j, w in enumerate(sent):
                pred = preds[j]
                corr = corrs[j]
                prob = probs[j]
                fw.write(("{} {} {} {}\n").format(w, corr, pred, prob))
            fw.write("\n")
