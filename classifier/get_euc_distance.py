from scipy.spatial import distance
import numpy as np

def common_word_EucSimilarity(label_mat_row, target_label_mat):
    cwProb=list()
    for row_vec in target_label_mat:
        cwProb.append(distance.euclidean( label_mat_row, row_vec ))
    return cwProb

def eucSim_mostCommonWords(label_mat, commonWords_mat):
    eucSim_commonWords=list()
    for row in label_mat:
        eucSim_commonWords.append(common_word_EucSimilarity(np.nan_to_num(row), commonWords_mat))

    return eucSim_commonWords
