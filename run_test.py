__author__ = 'iankuoli'

import numpy
import time
import sys
import subprocess
import os
import random
import functools
import gensim
import re
import logging
import theano

import load
import RNNModel
from accuracy import conlleval
from tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

    s = {'fold': 3, # 5 folds 0,1,2,3,4
         'lr': 0.07, #0.0627142536696559,
         'verbose': 1,
         'decay': False, # decay on the learning rate if improvement stops
         'win': 7, # number of words in the context window
         'bs': 9, # number of backprop through time steps
         'nhidden':1000, # number of hidden units
         'seed': 1976,
         'emb_dimension': 200, # dimension of word embedding
         'nepochs': 50}

    folder = 'run'
    # model feature dim has 200
    model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)

    # load the training dataset
    train_data = list()
    test_data = list()



    # parse each question's 5 possible answers `[...]` from test data
    setTestLabels = set()
    f_test = open('testing.txt', 'r')
    extract_test_answer_option = re.compile(r'\[(\w+)\]').search
    for line in f_test:
        # match_obj = re.search(r"\[.*\]", line)
        lines = line.strip('\n')[4:].split(' ')
        for word in lines:
            if word.isalnum():
                setTestLabels.add(word)

        match_obj = extract_test_answer_option(line)
        if match_obj:
            label_word = match_obj.group(1)
            setTestLabels.add(label_word)

    # label:
    # labelindx:
    # vec:
    label2vec = dict()
    vec2label = dict()
    labelindx2word = dict()
    word2labelindx = dict()
    label_indx = 0

    # add test labels as word
    # TODO: make the test labels stable (sort?)
    for label in setTestLabels:
        labelindx2word[label_indx] = label
        word2labelindx[label] = label_indx
        label_indx += 1


    # start word
    labelindx2word[label_indx] = "<s>"
    word2labelindx["<s>"] = label_indx

    # start word
    labelindx2word[label_indx+1] = "</s>"
    word2labelindx["</s>"] = label_indx+1

    # other word
    labelindx2word[label_indx+2] = "XXXXXX"
    word2labelindx["XXXXX"] = label_indx+2

    vocsize = len(model.vocab)
    nclasses = len(labelindx2word)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = RNNModel.RNNModel(nh=s['nhidden'],
                            nc=nclasses,
                            ne=vocsize,
                            de=s['emb_dimension'],
                            cs=7)
    rnn.load(folder)
    # read in test file
    test_questions = []
    with open('testing_data.cleaned.txt') as f:
        for answer_option in f:
            test_questions.append(
                list(answer_option.rstrip().split(' '))
            )
    # loop by question
    for answer_option in test_questions:
        x_fvec = []
        labels = []
        termss = []
        prediction_test = []
        for term in answer_option:
            # convert word to feature vector
            if term in model:
                x_fvec.append(model[term])
            else:
                # for instance: 'good-humoured' ==> 'good'
                try:
                    x_fvec.append(model[term.split('-')[0]])
                except KeyError:
                    # same hack as in the training process
                    x_fvec.append(model['some'])
            # map word to label_index
            if term in word2labelindx:
                # is label
                labels.append(word2labelindx[term])
            else:
                # not a label
                labels.append(word2labelindx["XXXXX"])
            termss.append(term)

            cwords = contextwin(x_fvec, s['win'], model["<s>"], model["</s>"])
            predictions_test.append(rnn.test(numpy.asarray(cwords).astype('float32')))
