#! /usr/bin/env python
'''
train_higgs.py tests discriminations of Higgs and non-Higgs events

Author: Alejandro M. Sanchez
Institution: Florida State University
Data Last Edited: April 19, 2017
'''
import random, time
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve,auc

#files used
higgs_train = 'higgs_train.dat'
bkg_train = 'bkg_train.dat'
higgs_val = 'higgs_val.dat'
bkg_val = 'bkg_val.dat'
cms_file = 'data.dat'

#mlp initial parameters
params = {'hidden_layer_sizes' : (100,50),
          'activation' : 'tanh',
          'solver' : 'adam',
          'early_stopping' : False,
          'validation_fraction' : 0.1,
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

    #training files
    training_data_1 = []
    training_data_real = []
    training_data_all = []
    f_hig = open(higgs_train, 'r')
    f_bkg = open(bkg_train, 'r')
    hig_lines_t = f_hig.readlines()
    bkg_lines_t = f_bkg.readlines()
    f_hig.close()
    f_bkg.close()

    #validation files
    validation_data = []
    f_hig = open(higgs_val, 'r')
    f_bkg = open(bkg_val, 'r')
    hig_lines_v = f_hig.readlines()
    bkg_lines_v = f_bkg.readlines()
    f_hig.close()
    f_bkg.close()

    #find out number of events being used for each style of training
    nTrain_1  = 2*min(len(hig_lines_t)-1, len(bkg_lines_t)-1)
    nTrain_real = len(hig_lines_t) - 1 + len(bkg_lines_t) - 1
    print '\nhiggs training file: ' + higgs_train
    print 'bkg training file: ' +  bkg_train
    print 'number of training events: '
    print '\tSNR = 1: ' + str(nTrain_1) + ' events'
    print '\tRealistic SNR: ' + str(nTrain_1) + ' events'
    print '\tAll Data:' + str(nTrain_real) + ' events'

    #using weighted validation data
    nVal  = len(hig_lines_v)-1 + len(bkg_lines_v)-1
    print '\nhiggs validation file: ' + higgs_val
    print 'bkg validation file: ' +  bkg_val
    print 'number of validation events: ' + str(nVal)

    #get higgs training data
    for line in hig_lines_t[1:]:
        temp = [float(x) for x in line.split()]
        training_data_all.append((temp,1))
        if len(training_data_1) < nTrain_1/2:
            training_data_1.append((temp,1))

    #get bkg training data
    for line in bkg_lines_t[1:]:
        temp = [float(x) for x in line.split()]
        training_data_all.append((temp,0))
        if len(training_data_1) < nTrain_1:
            training_data_1.append((temp,0))

    random.shuffle(training_data_all)
    random.shuffle(training_data_1)
    training_data_real = training_data_all[:nTrain_1]
    random.shuffle(training_data_real)

    #get higgs validation data
    for line in hig_lines_v[1:]:
        temp = [float(x) for x in line.split()]
        validation_data.append((temp,1))

    #get bkg validation data
    for line in bkg_lines_v[1:]:
        temp = [float(x) for x in line.split()]
        validation_data.append((temp,0))

    #separate training inputs and targets
    train_in_1 = []
    train_tar_1 = []
    for x,y in training_data_1:
        train_in_1.append(x[1:13])
        train_tar_1.append(y)

    train_in_real = []
    train_tar_real = []
    for x,y in training_data_real:
        train_in_real.append(x[1:13])
        train_tar_real.append(y)

    train_in_all = []
    train_tar_all = []
    for x,y in training_data_all:
        train_in_all.append(x[1:13])
        train_tar_all.append(y)

    val_w = []
    val_in = []
    val_tar = []
    for x,y in validation_data:
        val_w.append(x[0])
        val_in.append(x[1:13])
        val_tar.append(y)

#-------------------------------------------------------------------------------

    #train SNR = 1 mlp
    print '\ntraining SNR = 1 mlp'
    t_init = time.time()
    mlp_1 = MLPClassifier(**params)
    mlp_1.fit(train_in_1,train_tar_1)
    print 'training time: {0:0.3f} seconds.'.format(time.time()-t_init)

    #train Realistic SNR mlp (same amount of data as mlp above)
    print '\ntraining Realistic SNR mlp'
    t_init = time.time()
    mlp_real = MLPClassifier(**params)
    mlp_real.fit(train_in_real,train_tar_real)
    print 'training time: {0:0.3f} seconds.'.format(time.time()-t_init)

    #train with all data
    print '\ntraining with all data'
    t_init = time.time()
    mlp_all = MLPClassifier(**params)
    mlp_all.fit(train_in_all,train_tar_all)
    print 'training time: {0:0.3f} seconds.'.format(time.time()-t_init)

    #get validation scores for each mlp
    score_1 = mlp_1.score(val_in, val_tar, val_w)
    score_real = mlp_real.score(val_in, val_tar,val_w)
    score_all = mlp_all.score(val_in, val_tar, val_w)
    print '\nvalidation scores:'
    print '\tSNR = 1: ' + str(score_1)
    print '\tRealistic: ' + str(score_real)
    print '\tAll Data: ' + str(score_all)

    #get probabilties for being a signal event
    prob_1 = mlp_1.predict_proba(val_in)[:,1]
    prob_real = mlp_real.predict_proba(val_in)[:,1]
    prob_all = mlp_all.predict_proba(val_in)[:,1]

#-------------------------------------------------------------------------------
    #plotting

    #get false positive rate and true positive rate for roc curves
    fpr_1, tpr_1, thresh_1 = roc_curve(val_tar, prob_1,
                                             sample_weight=val_w)
    fpr_real, tpr_real, thresh_real = roc_curve(val_tar,prob_real,
                                             sample_weight=val_w)
    fpr_all, tpr_all, thresh_all = roc_curve(val_tar, prob_all,
                                             sample_weight=val_w)

    #get area under roc curves
    auc_1 = auc(fpr_1, tpr_1,reorder=True)
    auc_real = auc(fpr_real, tpr_real,reorder=True)
    auc_all = auc(fpr_all, tpr_all,reorder=True)

    #plot roc curves
    plt.figure()
    plt.plot(fpr_1,tpr_1,label='SNR = 1: '
            'auc = {0:0.3f}, score = {1:0.3f}'.format(auc_1,score_1))
    plt.plot(fpr_real,tpr_real,label='Realistic SNR: '
            'auc = {0:0.3f}, score = {1:0.3f}'.format(auc_real,score_real))
    plt.plot(fpr_all, tpr_all, label= 'All Data: '
            'auc = {0:0.3f}, score = {1:0.3f}'.format(auc_all,score_all))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('Training of Higgs vs non-Higgs events with various SNR:\n'
    'Using Weighted Data')
    plt.savefig('roc_curve_higgs.png')

#-------------------------------------------------------------------------------

    #get cms data
    f_dat = open(cms_file,'r')
    dat_lines = f_dat.readlines()
    f_dat.close()

    cms_data = []

    for line in dat_lines[1:]:
        temp = [float(x) for x in line.split()]
        cms_data.append(temp[1:13])

    #discriminate cms data
    cms_prob = mlp_all.predict_proba(cms_data)
    cms_pred = [prob[1] for prob in cms_prob]

    #plot histogram of signal probabilties for cms data
    plt.figure()
    plt.hist(cms_pred,range=(0,1),bins=50)
    plt.xlabel('Probality of Being a Signal Event')
    plt.ylabel('Counts')
    plt.title('Higgs vs non-Higgs Discrimination\n'
              'Probility Distributions for CMS Data')
    plt.savefig('cmsplot.png')

#-------------------------------------------------------------------------------

    #get signal probabilties for validation data set, and split into sets of
    #true signal and background
    pred_hig = []
    hig_w = []
    pred_bkg = []
    bkg_w = []

    for x in range(len(val_in)):
        if val_tar[x] == 0:
            pred_bkg.append(prob_all[x])
            bkg_w.append(val_w[x])
        elif val_tar[x] == 1:
            pred_hig.append(prob_all[x])
            hig_w.append(val_w[x])

    #plot histograms of signal probabilties for signal and background events
    #in validation data set
    plt.figure()
    plt.hist((pred_bkg,pred_hig),range=(0,1),bins=50,weights=(bkg_w,hig_w),
            label=('Background','Signal'))
    plt.title('Higgs vs non-Higgs Discrimination\n'
              'Probility Distributions for Validation Data')
    plt.xlabel('Probality of Being a Signal Event')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig('pred_hist.png')

    #same histograms but normalized
    plt.figure()
    plt.hist((pred_bkg,pred_hig),range=(0,1),bins=50,weights=(bkg_w,hig_w),
             normed=True,label=('Background','Signal'))
    plt.title('Higgs vs non-Higgs Discrimination\n'
              'Normalized Probility Distributions for Validation Data')
    plt.xlabel('Probality of Being a Signal Event')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig('pred_hist_norm.png')

    plt.show()

    print '\n\tAu Revoir!\n'

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

try:
    main()
except KeyboardInterrupt:
    print '\n\tok bye\n'
