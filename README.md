# vbf-train
Programs that I used during my honors thesis to produce my results. Demonstrates effective discrimination of data using raw variables instead of cleverly derived ones by using a complex neural network with two hidden layers and a stochastic optimizer algorithm called Adam.

Programs ending in .dat are the data files used for training and validation. Those with the uw prefix are unweighted versions of the others, with all weights rescaled to 1.

Due to the size of one of the files, it had to be compressed in order to be uploaded to github. Please uncompress before using.

To run any of the programs, ensure that all data is unpacked and run $ python [filename]

All programs require 3rd party libraries sklearn (http://scikit-learn.org) and matplotlib (http://matplotlib.org)

    train_higgs.py tests discriminations of Higgs and non-Higgs events.

    train_vbf.py tests discriminations of vector boson fusion (VBF) Higgs events and non-VBF events.

    train_all.py combines both discrinants to produced 2D histrograms of the results.
