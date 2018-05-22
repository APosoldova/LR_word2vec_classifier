import numpy as np
from get_euc_distance import common_word_EucSimilarity

def get_label_probability(feature_matrix, coefficients):
    labelProb = np.dot(feature_matrix, coefficients)
    return labelProb

def get_probability(wv_bio, wv_label, followerCount_test, coefficientsLO):
    #compute similarities of test item and label common words
    eucSim_bio_commonWords=common_word_EucSimilarity(wv_bio, wv_label)
    eucSim_bio_commonWords=np.asarray(eucSim_bio_commonWords).reshape(1,len(eucSim_bio_commonWords))
    
    #create feature matrix
    feature_matrix_test=np.concatenate((eucSim_bio_commonWords, followerCount_test), axis=1)
    #compute probabilites
    labelProbLO=get_label_probability(feature_matrix_test, coefficientsLO)
    return labelProbLO
