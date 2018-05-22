import pickle
from classifier.get_euc_distance import common_word_EucSimilarity

def test_euc_sim():
    wv_label=pickle.load(open('classifier/pcl_files/wv_30commonInfluencer_mat.pkl', 'rb'))
    wv_bio=pickle.load(open('pcl_files/test_bio_wvMat.pkl', 'rb'))
    res = common_word_EucSimilarity(wv_bio, wv_label)
    exp_res=pickle.load(open('classifier/pcl_files/test_euc_dist_test.pkl', 'rb'))
    assert res==exp_res
