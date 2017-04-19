#! /usr/bin/env python
'''
train_vbf.py tests discriminations of vector boson fusion (VBF) Higgs events
and non-VBF events

Author: Alejandro M. Sanchez
Institution: Florida State University
Data Last Edited: April 19, 2017
'''
import random
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve,auc

higgs_train = 'higgs_train.dat'
higgs_val = 'higgs_val.dat'

params = {'hidden_layer_sizes' : (100,50),
          'activation' : 'tanh',
          'solver' : 'adam',
          'early_stopping' : False,
          'verbose' : True}

#Fields contained in entries, for reference
'''
FIELDS = ['f_weight',       0
          'f_lept1_pt',     1
          'f_lept1_eta',    2
          'f_lept1_phi',    3
          'f_lept2_pt',     4
          'f_lept2_eta',    5
          'f_lept2_phi',    6
          'f_lept3_pt',     7
          'f_lept3_eta',    8
          'f_lept3_phi',    9
          'f_lept4_pt',     10
          'f_lept4_eta',    11
          'f_lept4_phi',    12
          'f_jet1_pt',      13
          'f_jet1_eta',     14
          'f_jet1_phi',     15
          'f_jet2_pt',      16
          'f_jet2_eta',     17
          'f_jet2_phi']     18

          'f_deltajj',      19    #just entries in higgs files
          'f_massjj',       20
          'f_sample']       21
'''

#-------------------------------------------------------------------------------

def main():

    #get info from files

    #training data file
    training_data = []
    f_hig = open(higgs_train, 'r')
    hig_lines_t = f_hig.readlines()
    f_hig.close()

    #validation data file
    validation_data = []
    f_hig = open(higgs_val, 'r')
    hig_lines_v = f_hig.readlines()
    f_hig.close()

    print '\ntraining file: ' + higgs_train
    print 'validation file ' + higgs_val

    #get higgs training data
    for line in hig_lines_t[1:]:
        temp = [float(x) for x in line.split()]
        if temp[-1] == 1:
            training_data.append((temp,1))
        else:
            training_data.append((temp,0))

    #get higgs validation data
    for line in hig_lines_v[1:]:
        temp = [float(x) for x in line.split()]
        if temp[-1] == 1:
            validation_data.append((temp,1))
        else:
            validation_data.append((temp,0))

    print '\ntotal num train events = ' + str(len(training_data))
    print 'total num val events = ' + str(len(validation_data))

    #only keep events which contain two jets
    training_data_red = []
    for entry in training_data:
        if entry[0][16] >= 0:
            training_data_red.append(entry)

    validation_data_red = []
    for entry in validation_data:
        if entry[0][16] >= 0:
            validation_data_red.append(entry)

    training_data = training_data_red
    validation_data = validation_data_red

    #count number of events being used
    print '\nreduced num train events: ' + str(len(training_data))
    print 'reduced num val events: ' + str(len(validation_data))

    num_sig = 0
    num_bkg = 0
    for entry in training_data:
        if entry[1] == 1:
            num_sig += 1
        else:
            num_bkg += 1

    print '\nnum train sig events: ' + str(num_sig)
    print 'num train bkg events: ' + str(num_bkg)

    num_sig = 0
    num_bkg = 0
    for entry in validation_data:
        if entry[1] == 1:
            num_sig += 1
        else:
            num_bkg += 1

    print '\nnum val sig events: ' + str(num_sig)
    print 'num val bkg events: ' + str(num_bkg)

    #shuffle training data, get targets and validation weights
    random.shuffle(training_data)
    train_tar = [y for x,y in training_data]
    val_w = [x[0] for x,y in validation_data]
    val_tar = [y for x,y in validation_data]

#-------------------------------------------------------------------------------
    #train mlp's and get signal probabilties

    #train with jet and lepton 4 vectors
    train_in = [x[1:19] for x,y in training_data]
    val_in = [x[1:19] for x,y in validation_data]
    mlp_jl = MLPClassifier(**params)
    mlp_jl.fit(train_in, train_tar)
    score_jl = mlp_jl.score(val_in, val_tar, val_w)
    prob_jl = mlp_jl.predict_proba(val_in)[:,1]

    #train with just lepton 4 vectors
    train_in = [x[1:13] for x,y in training_data]
    val_in = [x[1:13] for x,y in validation_data]
    mlp_lep = MLPClassifier(**params)
    mlp_lep.fit(train_in, train_tar)
    score_lep = mlp_lep.score(val_in, val_tar, val_w)
    prob_lep = mlp_lep.predict_proba(val_in)[:,1]

    #train with just jet 4 vectors
    train_in = [x[13:19] for x,y in training_data]
    val_in = [x[13:19] for x,y in validation_data]
    mlp_jet = MLPClassifier(**params)
    mlp_jet.fit(train_in, train_tar)
    score_jet = mlp_jet.score(val_in, val_tar, val_w)
    prob_jet = mlp_jet.predict_proba(val_in)[:,1]

    #train with deltjj and massjj
    train_in = [x[19:21] for x,y in training_data]
    val_in = [x[19:21] for x,y in validation_data]
    mlp_dm = MLPClassifier(**params)
    mlp_dm.fit(train_in, train_tar)
    score_dm = mlp_dm.score(val_in, val_tar, val_w)
    prob_dm = mlp_dm.predict_proba(val_in)[:,1]

#-------------------------------------------------------------------------------
    #plotting

    #get false positive rate and true positive rate for roc curves
    fpr_jl, tpr_jl, thresh_jl = roc_curve(val_tar, prob_jl,
                                          sample_weight=val_w)
    fpr_lep, tpr_lep, thresh_lep = roc_curve(val_tar, prob_lep,
                                          sample_weight=val_w)
    fpr_jet, tpr_jet, thresh_jet = roc_curve(val_tar, prob_jet,
                                          sample_weight=val_w)
    fpr_dm, tpr_dm, thresh_dm = roc_curve(val_tar, prob_dm,
                                          sample_weight=val_w)

    #get area under roc curves
    auc_jl = auc(fpr_jl,tpr_jl,reorder=True)
    auc_lep = auc(fpr_lep,tpr_lep,reorder=True)
    auc_jet = auc(fpr_jet,tpr_jet,reorder=True)
    auc_dm = auc(fpr_dm,tpr_dm,reorder=True)

    #plot roc curves
    plt.figure()
    plt.plot(fpr_jl,tpr_jl,label='jet and lepton 4 vectors'
            '\nauc = {0:0.3f}, score = {1:0.3f}'.format(auc_jl, score_jl))
    plt.plot(fpr_lep,tpr_lep,label='just lepton 4 vectors'
            '\nauc = {0:0.3f}, score = {1:0.3f}'.format(auc_lep, score_lep))
    plt.plot(fpr_jet,tpr_jet,label='just jet 4 vectors'
            '\nauc = {0:0.3f}, score = {1:0.3f}'.format(auc_jet, score_jet))
    plt.plot(fpr_dm,tpr_dm,label='deltjj and massjj'
            '\nauc = {0:0.3f}, score = {1:0.3f}'.format(auc_dm, score_dm))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('Training VBF discrimination using various input variables')
    plt.savefig('roc_curve_vbf.png')
    plt.show()

    print '\n\tAu Revoir!\n'

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

try:
    main()
except KeyboardInterrupt:
    print '\n\tok bye\n'
