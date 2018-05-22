import classifier.process_test_bio
import nltk
import pickle

def test_stop_words():
    stopwords = nltk.corpus.stopwords.words('english')
    bio="And they left so the room was empty"
    exp_res = ['left', 'room', 'empty']
    res=process_test_bio.remove_stop_words(bio.split(),stopwords)
    assert res == exp_res

def test_reg_exp():
    bio="There was $700/10=70"
    exp_res='There was  700 10 70'
    res=process_test_bio.remove_regexp(bio)
    assert res == exp_res

def test_wvMat():
    Google_model = gensim.models.KeyedVectors.load_word2vec_format('classifier/pcl_files/GoogleNews-vectors-negative300.bin', binary=True)
    bio = 'Sustainability Management Professional Founder of @BackboneMS \n#Sustainability #Management #System QHSE, Benchmark, PDCA, Intranet, #SaaS, #Startup, #Blogger.'
    res= process_test_bio.create_test_wvMat(bio,Google_model, 300)
    exp_res = pickle.load(open('classifier/pcl_files/test_bio_wvMat.pkl', 'rb'))
    assert exp_res == res

