#! /usr/bin/env python
'''
train_all.py combines both discrinants to produced 2D histrograms of the results

Author: Alejandro M. Sanchez
Institution: Florida State University
Data Last Edited: April 19, 2017
'''
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

          'f_deltajj',      19    #just entries in higgs and cms files
          'f_massjj',       20
          'f_sample']       21
'''

#-------------------------------------------------------------------------------

def main():

    #get info from files

    #training files
    training_data = []
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

    #cms data file
    cms_data = []
    f_cms = open(cms_file, 'r')
    cms_lines = f_cms.readlines()
    f_cms.close()

    #find out number of events being used for each style of training
    nTrain = len(hig_lines_t) - 1 + len(bkg_lines_t) - 1
    print '\nhiggs training file: ' + higgs_train
    print 'bkg training file: ' +  bkg_train
    print 'number of training events: ' + str(nTrain)

    nVal  = len(hig_lines_v)-1 + len(bkg_lines_v)-1
    print '\nhiggs validation file: ' + higgs_val
    print 'bkg validation file: ' +  bkg_val
    print 'number of validation events: ' + str(nVal)

    nCMS = len(cms_lines)-1
    print "\ncms file: " + cms_file
    print "number of training events: " + str(nCMS) + '\n'

    #get higgs training data
    for line in hig_lines_t[1:]:
        temp = [float(x) for x in line.split()]
        if temp[-1] == 1:
            training_data.append((temp,1,1))
        else:
            training_data.append((temp,1,0))

    #get bkg training data
    for line in bkg_lines_t[1:]:
        temp = [float(x) for x in line.split()]
        training_data.append((temp,0,0))

    #get higgs validation data
    for line in hig_lines_v[1:]:
        temp = [float(x) for x in line.split()]
        if temp[-1] == 1:
            validation_data.append((temp,1,1))
        else:
            validation_data.append((temp,1,0))

    #get bkg validation data
    for line in bkg_lines_v[1:]:
        temp = [float(x) for x in line.split()]
        validation_data.append((temp,0,0))

    #get cms data
    for line in cms_lines[1:]:
        temp = [float(x) for x in line.split()]
        cms_data.append(temp)

    #shuffle training data
    random.shuffle(training_data)

    #separate training inputs and targets
    train_in_hig = []
    train_in_vbf = []
    train_tar_hig = []
    train_tar_vbf = []
    for x,y,z in training_data:
        train_in_hig.append(x[1:13])
        train_tar_hig.append(y)
        if x[16] >= 0:
            train_in_vbf.append(x[1:19])
            train_tar_vbf.append(z)

    val_w = []
    val_in_hig = []
    val_in_vbf = []
    val_tar_hig = []
    val_tar_vbf = []
    for x,y,z in validation_data:
        if x[16] >= 0:
            val_w.append(x[0])
            val_in_hig.append(x[1:13])
            val_tar_hig.append(y)
            val_in_vbf.append(x[1:19])
            val_tar_vbf.append(z)

    cms_in_hig = []
    cms_in_vbf = []
    for x in cms_data:
        if x[16] >= 0:
            cms_in_hig.append(x[1:13])
            cms_in_vbf.append(x[1:19])

    print 'num higgs training data: ' + str(len(train_in_hig))
    print 'num vbf training data: ' + str(len(train_in_vbf))
    print 'num validation data: ' + str(len(val_w))
    print 'num cms data = ' + str(len(cms_in_hig)) + '\n'

#-------------------------------------------------------------------------------
    #set up and train mlp's

    #Training Higgs classification
    mlp_hig = MLPClassifier(**params)
    mlp_hig.fit(train_in_hig, train_tar_hig)

    #Training for vbf classification
    mlp_vbf = MLPClassifier(**params)
    mlp_vbf.fit(train_in_vbf, train_tar_vbf)

    val_prob_hig = mlp_hig.predict_proba(val_in_hig)[:,1]
    val_prob_vbf = mlp_vbf.predict_proba(val_in_vbf)[:,1]
    cms_prob_hig = mlp_hig.predict_proba(cms_in_hig)[:,1]
    cms_prob_vbf = mlp_vbf.predict_proba(cms_in_vbf)[:,1]

    #Separate validation events and plot 2d histograms for each set
    x_bkg = []
    y_bkg = []
    w_bkg = []
    x_hig = []
    y_hig = []
    w_hig = []
    x_vbf = []
    y_vbf =[]
    w_vbf = []
    for x in range(len(val_w)):
        if val_tar_hig[x] == 0:
            x_bkg.append(val_prob_hig[x])
            y_bkg.append(val_prob_vbf[x])
            w_bkg.append(val_w[x])
        elif val_tar_hig[x] == 1 and val_tar_vbf[x] == 0:
            x_hig.append(val_prob_hig[x])
            y_hig.append(val_prob_vbf[x])
            w_hig.append(val_w[x])
        elif val_tar_hig[x] == 1 and val_tar_vbf[x] == 1:
            x_vbf.append(val_prob_hig[x])
            y_vbf.append(val_prob_vbf[x])
            w_vbf.append(val_w[x])

    x_hv = x_hig + x_vbf
    y_hv = y_hig + y_vbf
    w_hv = w_hig + w_vbf

    plt.figure()
    plt.subplot(321)
    plt.hist2d(val_prob_hig,val_prob_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]],
    weights=val_w)
    plt.title("All data")
    plt.subplot(322)
    plt.hist2d(x_bkg,y_bkg,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_bkg)
    plt.title("Background")
    plt.subplot(323)
    plt.hist2d(x_hig,y_hig,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_hig)
    plt.title("Non-VBF Higgs")
    plt.subplot(324)
    plt.hist2d(x_vbf,y_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_vbf)
    plt.title("VBF Higgs")
    plt.subplot(325)
    plt.hist2d(x_hv,y_hv,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_hv)
    plt.title("All Higgs events")
    plt.subplot(326)
    plt.hist2d(cms_prob_hig,cms_prob_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]])
    plt.title("CMS Data")
    plt.savefig("hist2d.png")

    plt.figure()
    plt.subplot(321)
    plt.hist2d(val_prob_hig,val_prob_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]],
    weights=val_w,norm=LogNorm())
    plt.title("All data - Log")
    plt.subplot(322)
    plt.hist2d(x_bkg,y_bkg,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_bkg,
    norm=LogNorm())
    plt.title("Background - Log")
    plt.subplot(323)
    plt.hist2d(x_hig,y_hig,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_hig,
    norm=LogNorm())
    plt.title("Non-VBF Higgs - Log")
    plt.subplot(324)
    plt.hist2d(x_vbf,y_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_vbf,
    norm=LogNorm())
    plt.title("VBF Higgs - Log")
    plt.hist2d(x_hv,y_hv,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_hv,
    norm=LogNorm())
    plt.subplot(325)
    plt.hist2d(x_hv,y_hv,bins=20,range=[[0.0,1.0],[0.0,1.0]],weights=w_hv,
    norm=LogNorm())
    plt.title("All Higgs Events - Log")
    plt.subplot(326)
    plt.hist2d(cms_prob_hig,cms_prob_vbf,bins=20,range=[[0.0,1.0],[0.0,1.0]],
    norm=LogNorm())
    plt.title("CMS Data - Log")
    plt.savefig("hist2dlog.png")

    plt.show()

    print '\n\tAu Revoir!\n'

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

try:
    main()
except KeyboardInterrupt:
    print '\n\tok bye\n'
