# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 09:34
# @Author  : Xiaoyu Xing
# @File    : feature_pu_model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch.autograd import Variable
import argparse
from utils.data_utils import DataPrepare
from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet, TimeDistributed
from progressbar import *


class PULSTMCNN(nn.Module):
    def __init__(self, dp, charModel, wordModel, caseModel, featureModel, inputSize, hiddenSize, layerNum, dropout):
        super(PULSTMCNN, self).__init__()
        self.dp = dp
        self.charModel = TimeDistributed(charModel, self.dp.char2Idx)
        self.wordModel = wordModel
        self.caseModel = caseModel
        self.featureModel = featureModel
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=layerNum,batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hiddenSize, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Softmax(dim=2)
            # nn.Linear(200, 1)
        )

    def forward(self, token, case, char, feature):
        charOut, sortedLen1, reversedIndices1 = self.charModel(char)
        wordOut, sortedLen2, reversedIndices2 = self.wordModel(token)
        caseOut, sortedLen3, reversedIndices3 = self.caseModel(case)
        featureOut, sortedLen4, reversedIndices4 = self.featureModel(feature)

        encoding = torch.cat([wordOut.float(), caseOut.float(), charOut.float(), featureOut.float()], dim=2)

        sortedLen = sortedLen1
        reverseIndices = reversedIndices1

        packed_embeds = pack_padded_sequence(encoding, sortedLen, batch_first=True)

        maxLen = sortedLen[0]
        mask = torch.zeros([len(sortedLen), maxLen, 2])
        for i, l in enumerate(sortedLen):
            mask[i][:l][:] = 1

        lstmOut, (h, _) = self.lstm(packed_embeds)

        paddedOut = pad_packed_sequence(lstmOut, batch_first=True)

        # print(paddedOut)

        fcOut = self.fc(paddedOut[0])

        fcOut = fcOut * mask.cuda()
        fcOut = fcOut[reverseIndices]

        return fcOut

    def loss_func(self, yTrue, yPred, type):
        y = torch.eye(2)[yTrue].float().cuda()
        if len(y.shape) == 1:
            y = y[None, :]
        # y = torch.from_numpy(yTrue).float().cuda()
        if type == 'bnpu' or type == 'bpu':
            loss = torch.mean((y * (1 - yPred)).sum(dim=1))
        elif type == 'upu':
            loss = torch.mean((-y * torch.log(yPred)).sum(dim=1))
        # loss = 0.5 * torch.max(1-yPred*(2.0*yTrue-1),0)
        return loss


class Trainer(object):
    def __init__(self, model, prior, beta, gamma, learningRate, m):
        self.model = model
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learningRate,
                                          weight_decay=1e-8)
        self.m = m
        self.prior = prior
        self.bestResult = 0
        self.beta = beta
        self.gamma = gamma
        self.positive = np.eye(2)[1]
        self.negative = np.eye(2)[0]

    def train_mini_batch(self, batch, args):
        token, case, char, feature, label, flag = batch
        length = [len(i) for i in flag]
        maxLen = max(length)
        fids = []
        lids = []
        for s in flag:
            f = list(s)
            f += [np.array([-1, -1]) for _ in range(maxLen - len(f))]
            fids.append(f)
        for s in label:
            l = list(s)
            l += [np.array([-1, -1]) for _ in range(maxLen - len(l))]
            lids.append(l)
        fids = np.array(fids)
        lids = np.array(lids)

        postive = (fids == self.positive) * 1
        unlabeled = (fids == self.negative) * 1

        self.optimizer.zero_grad()
        result = self.model(token, case, char, feature)

        hP = result.masked_select(torch.from_numpy(postive).byte().cuda()).contiguous().view(-1, 2)
        hU = result.masked_select(torch.from_numpy(unlabeled).byte().cuda()).contiguous().view(-1, 2)
        if len(hP) > 0:
            pRisk = self.model.loss_func(1, hP, args.type)
        else:
            pRisk = torch.FloatTensor([0]).cuda()
        uRisk = self.model.loss_func(0, hU, args.type)
        nRisk = uRisk - self.prior * (1 - pRisk)
        risk = self.m * pRisk + nRisk

        if args.type == 'bnpu':
            if nRisk < self.beta:
                risk = -self.gamma * nRisk
        # risk = self.model.loss_func(label, result)
        (risk).backward()
        self.optimizer.step()
        pred = torch.argmax(hU, dim=1)
        label = Variable(torch.LongTensor(list(lids))).cuda()
        unlabeledY = label.masked_select(torch.from_numpy(unlabeled).byte().cuda()).contiguous().view(-1, 2)

        acc = torch.mean((torch.argmax(unlabeledY, dim=1) == pred).float())
        return acc.item(), risk.item(), pRisk.item(), nRisk.item()

    def test(self, batch, length):
        token, case, char, feature = batch
        maxLen = max([x for x in length])
        mask = np.zeros([len(token), maxLen, 2])
        for i, x in enumerate(length):
            mask[i][:x][:] = 1
        result = self.model(token, case, char, feature)
        # print(result)
        result = result.masked_select(torch.from_numpy(mask).byte().cuda()).contiguous().view(-1, 2)
        pred = torch.argmax(result, dim=1)

        temp = result[:, 1]
        return pred.cpu().numpy(), temp.detach().cpu().numpy()

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)

    def decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1 + 0.05 * epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU NER")
    # data
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--drop_out', type=float, default=0.5)
    parser.add_argument('--m', type=float, default=0.3)
    parser.add_argument('--flag', default="PER")
    parser.add_argument('--dataset', default="conll2003")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--print_time', type=int, default=1)
    parser.add_argument('--pert', type=float, default=1.0)
    parser.add_argument('--type', type=str, default='bnpu')  # bpu upu

    args = parser.parse_args()

    dp = DataPrepare(args.dataset)
    mutils = FeaturedDetectionModelUtils(dp)

    trainSet, validSet, testSet, prior = mutils.load_dataset(args.flag, args.dataset, args.pert)

    trainSize = len(trainSet)
    validSize = len(validSet)
    testSize = len(testSet)
    print(("train set size: {}, valid set size: {}, test set size: {}").format(trainSize, validSize, testSize))

    charcnn = CharCNN(dp.char2Idx)
    wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
    casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
    featurenet = FeatureNet()
    pulstmcnn = PULSTMCNN(dp, charcnn, wordnet, casenet, featurenet, 150, 200, 1, args.drop_out)

    if torch.cuda.is_available:
        charcnn.cuda()
        wordnet.cuda()
        casenet.cuda()
        featurenet.cuda()
        pulstmcnn.cuda()
        torch.cuda.manual_seed(1013)

    trainer = Trainer(pulstmcnn, prior, args.beta, args.gamma, args.lr, args.m)

    time = 0

    bar = ProgressBar(maxval=int((len(trainSet) - 1) / args.batch_size))

    train_sentences = dp.read_origin_file("data/" + args.dataset + "/train.txt")
    trainSize = int(len(train_sentences) * args.pert)
    train_sentences = train_sentences[:trainSize]
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

    valid_sentences = dp.read_origin_file("data/" + args.dataset + "/valid.txt")
    valid_words = []
    valid_efs = []
    for s in valid_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        valid_words.append(temp)
        valid_efs.append(temp2)

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

    for e in range(1, 1000):
        print("Epoch: {}".format(e))
        bar.start()
        risks = []
        prisks = []
        nrisks = []
        for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch) in enumerate(
                mutils.iterateSet(trainSet, batchSize=args.batch_size, mode="TRAIN")):
            bar.update(step)
            batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch]
            acc, risk, prisk, nrisk = trainer.train_mini_batch(batch, args)
            risks.append(risk)
            prisks.append(prisk)
            nrisks.append(nrisk)
        meanRisk = np.mean(np.array(risks))
        meanRisk2 = np.mean(np.array(prisks))
        meanRisk3 = np.mean(np.array(nrisks))
        print("risk: {}, prisk: {}, nrisk: {}".format(meanRisk, meanRisk2, meanRisk3))

        if e % 5 == 0:
            trainer.decay_learning_rate(e, args.lr)
        if e % args.print_time == 0:
            # train set
            pred_train = []
            corr_train = []
            for step, (x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch,
                       y_train_batch) in enumerate(
                mutils.iterateSet(trainSet, batchSize=100, mode="TEST", shuffle=False)):
                trainBatch = [x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch]
                correcLabels = []
                for x in y_train_batch:
                    for xi in x:
                        correcLabels.append(xi)
                lengths = [len(x) for x in x_word_train_batch]
                predLabels, _ = trainer.test(trainBatch, lengths)
                correcLabels = np.array(correcLabels)
                # print(predLabels)
                # print(correcLabels)
                assert len(predLabels) == len(correcLabels)

                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = correcLabels[start:end]
                    pred_train.append(p)
                    corr_train.append(c)
                    start = end

            # prec, rec, f1 = dp.evaluate_ner_tagging("data/" + args.dataset + "/train.txt",pred_train)

            newSentences = []
            for i, s in enumerate(train_words):
                sent = []
                assert len(s) == len(train_efs[i]) == len(pred_train[i])
                for j, item in enumerate(s):
                    sent.append([item, train_efs[i][j], pred_train[i][j]])
                newSentences.append(sent)

            newSentences_, newLabels, newPreds = dp.wordLevelGeneration(newSentences)
            p_train, r_train, f1_train = dp.compute_precision_recall_f1(newLabels, newPreds, args.flag, 1)
            print("Precision: {}, Recall: {}, F1: {}".format(p_train, r_train, f1_train))

            # valid set
            pred_valid = []
            corr_valid = []
            for step, (
                    x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
                    y_test_batch) in enumerate(
                mutils.iterateSet(validSet, batchSize=100, mode="TEST", shuffle=False)):
                validBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
                correcLabels = []
                for x in y_test_batch:
                    for xi in x:
                        correcLabels.append(xi)
                lengths = [len(x) for x in x_word_test_batch]
                predLabels, _ = trainer.test(validBatch, lengths)
                correcLabels = np.array(correcLabels)
                assert len(predLabels) == len(correcLabels)

                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = correcLabels[start:end]
                    pred_valid.append(p)
                    corr_valid.append(c)
                    start = end

            newSentencesValid = []
            for i, s in enumerate(valid_words):
                sent = []
                assert len(s) == len(valid_efs[i]) == len(pred_valid[i])
                for j, item in enumerate(s):
                    sent.append([item, valid_efs[i][j], pred_valid[i][j]])
                newSentencesValid.append(sent)

            newSentencesValid_, newLabelsValid, newPredsValid = dp.wordLevelGeneration(newSentencesValid)
            p_valid, r_valid, f1_valid = dp.compute_precision_recall_f1(newLabelsValid, newPredsValid, args.flag,
                                                                        1)
            print("Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))

            if f1_valid <= trainer.bestResult:
                time += 1
            else:
                trainer.bestResult = f1_valid
                time = 0
                trainer.save(
                    ("saved_model/{}_{}_{}_lr_{}_prior_{}_beta_{}_gamma_{}_percent_{}").format(args.type, args.dataset,
                                                                                               args.flag,
                                                                                               trainer.learningRate,
                                                                                               trainer.m,
                                                                                               trainer.beta,
                                                                                               trainer.gamma,
                                                                                               args.pert))
            if time > 5:
                print(("BEST RESULT ON VALIDATE DATA:{}").format(trainer.bestResult))
                break


    pulstmcnn.load_state_dict(
        torch.load(
            "saved_model/{}_{}_{}_lr_{}_prior_{}_beta_{}_gamma_{}_percent_{}".format(args.type, args.dataset, args.flag,
                                                                                     trainer.learningRate,
                                                                                     trainer.m,
                                                                                     trainer.beta,
                                                                                     trainer.gamma, args.pert)))

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
        predLabels, _ = trainer.test(testBatch, lengths)
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
    print("Test Result: Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))
