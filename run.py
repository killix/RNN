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
         'emb_dimension': 100, # dimension of word embedding
         'nepochs': 50}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # model feature dim has 200
    model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)

    # load the training dataset
    train_data = list()
    valid_data = list()
    test_data = list()

    # parse each question's 5 possible answers `[...]` from test data
    setTestLabels = set()
    f_test = open('testing.txt', 'r')
    for line in f_test:
        match_obj = re.search(r"\[.*\]", line)
        if match_obj:
            label_word = match_obj.group()
            setTestLabels.add(label_word[4:-5])

    # label:
    # labelindx:
    # vec:
    label2vec = dict()
    vec2label = dict()
    labelindx2word = dict()
    word2labelindx = dict()
    label_indx = 0

    for x in setTestLabels:
        labelindx2word[label_indx] = x
        word2labelindx[x] = label_indx
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


    f_train = open('try_it.txt', 'r')


    for sentence in f_train:
        train_data.append(sentence)

    # 1-5000 sentences are for validation
    valid_data = train_data[1:5000]

    # 5001- sentences are for training
    train_data = test_data[5001:]

    vocsize = len(model.vocab)
    nclasses = len(labelindx2word)
    nsentences = len(train_data)


    #
    #  --- sample code start ---
    #
    """
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word  = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y  = test_set

    vocsize = len(set(functools.reduce(lambda x, y: list(x) + list(y), train_lex + valid_lex + test_lex)))
    nclasses = len(set(functools.reduce(lambda x, y: list(x) + list(y), train_y + test_y + valid_y)))
    nsentences = len(train_lex)
    """

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    """
    (de * cs) -> nh -> nc
    nh :: dimension of the hidden layer
    nc :: number of classes
    ne :: number of word embeddings in the vocabulary
    de :: dimension of the word embeddings
    cs :: word window context size

    ce :: current epoch no
    """

    rnn = RNNModel.RNNModel(nh=s['nhidden'],
                            nc=nclasses,
                            ne=vocsize,
                            de=s['emb_dimension'],
                            cs=7)

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in range(s['nepochs']):
        # shuffle
        #shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in range(nsentences):
            # convert the i-th sentence to window with size s['win']

            #  --- start modified ---
            x_fvec = [] # x's feature vector of a sentence
            labels = [] # label list of a sentence
            for term in train_data[i]:#train_lex[i]:
                # convert word to feature vector
                if term in model:
                    x_fvec.append(model[term])
                else:
                    # for instance: 'good-humoured' ==> 'good'
                    x_fvec.append(model[term.split('-')[0]])

                # map word to label_index
                if term in word2labelindx:
                    # is label
                    labels.append(word2labelindx[term])
                else:
                    # not a label
                    labels.append(word2labelindx["XXXXX"])

            # add a PADDING-END word at the rightend (the last word in the sentence)
            labels.append(model["</s>"])
            # remove a PADDING_START word at the begining.
            labels.pop(0)
            #  --- end modified ---

            #cwords = contextwin(train_lex[i], s['win'])
            cwords = contextwin(x_fvec, s['win'], model["<s>"], model["</s>"])

            words  = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cwords, s['bs']))
            #labels = train_y[i]


            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])
                #rnn.normalize()
            if s['verbose']:
                print('[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),)
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        """
        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in test_lex ]
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in valid_lex ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]
        """

        predictions_test = [ map(lambda x: labelindx2word[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in test_data ]
        #groundtruth_test = [ map(lambda x: labelindx2word[x], y) for y in test_y ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_data]

        #predictions_valid = [ map(lambda x: labelindx2word[x], \
        #                     rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
        #                     for x in valid_data ]
        #groundtruth_valid = [ map(lambda x: labelindx2word[x], y) for y in valid_y ]
        #words_valid = [ map(lambda x: idx2word[x], w) for w in valid_data]

        predictions_valid = []
        groundtruth_valid = []

        for i in range(len(valid_data)):#train_lex[i]:

            x_fvec = [] # x's feature vector of a sentence
            labels = [] # label list of a sentence

            for term in valid_data[i]:
                # convert word to feature vector
                if term in model:
                    x_fvec.append(model[term])
                else:
                    # for instance: 'good-humoured' ==> 'good'
                    x_fvec.append(model[term.split('-')[0]])
                # map word to label_index
                if term in word2labelindx:
                    # is label
                    labels.append(word2labelindx[term])
                else:
                    # not a label
                    labels.append(word2labelindx["XXXXX"])

        #cwords = contextwin(train_lex[i], s['win'])
        cwords = contextwin(x_fvec, s['win'], model["<s>"], model["</s>"])
        predictions_valid.append(rnn.classify(numpy.asarray(cwords)))

        # add a PADDING-END word at the rightend (the last word in the sentence)
        labels.append(model["</s>"])
        # remove a PADDING_START word at the begining.
        labels.pop(0)

        # evaluation // compute the accuracy using conlleval.pl
        #res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        #res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        '''
        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if s['verbose']:
                print ('NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20)
            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            s['be'] = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ('')
        '''

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break

    #print('BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder)
