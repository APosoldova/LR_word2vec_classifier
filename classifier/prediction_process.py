from flask import Flask
from flask import request
import json
import os
import pickle
import gensim, logging
import nltk
import numpy as np
from process_test_bio import create_wvMat as create_test_wvMat
from get_prediction import get_probability
from get_env import get_env
import logging
from logging import StreamHandler

app = Flask(__name__)

#pyenv = get_env("PY_ENV")
pyenv= "dev"

# create logger
logger = logging.getLogger('simple_example')
file_handler = StreamHandler()
logger.setLevel(logging.DEBUG)  # set the desired logging level here
logger.addHandler(file_handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load Google news model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
Google_model = gensim.models.KeyedVectors.load_word2vec_format('pcl_files/GoogleNews-vectors-negative300.bin', binary=True) 

#nltk libraries
app.logger.info('Installing NLTK models')
if 'stopwords' in os.listdir( nltk.data.find("corpora")):
    app.logger.info('-- Stopwords corpora already downloaded')
else:
    app.logger.info('-- Downloading stopwords corpora')
    nltk.download('stopwords')

if 'punkt' in os.listdir(nltk.data.find("tokenizers")):
    app.logger.info('-- Punctuation corpora already downloaded')
else:
    app.logger.info('-- Downloading punctuation corpora')
nltk.download('punkt')



@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if pyenv == "dev" or pyenv=="production": 
        labels_dic={0: 'influencer', 1: 'brand', 2: 'news and media'}
           
        app.logger.info('Starting processing input')
        user_data = request.get_json(force=True)
        assert len(user_data['bio']) != 0, 'No bio provided'
        assert len(user_data['follower_count']) != 0, 'No follower count provided'

        processed_bio=create_test_wvMat(user_data,Google_model, 300)
        followerCount_test=np.asarray(int(user_data['follower_count'])).reshape(1,1)

        #load picke files
        logger.info("Loading pickle files...")
    
        coefficients_LO_Influencers = pickle.load(open('pcl_files/coefficients_LO_Influencers.pkl', 'rb'))
        coefficients_LO_brand = pickle.load(open('pcl_files/coefficients_LO_brand.pkl', 'rb'))
        coefficients_LO_NewsMedia = pickle.load(open('pcl_files/coefficients_LO_NewsMedia.pkl', 'rb'))

        wv_30commonInfluencer_mat = pickle.load(open('pcl_files/wv_30commonInfluencer_mat.pkl', 'rb'))
        wv_30commonBrand_mat = pickle.load(open('pcl_files/wv_30commonBrand_mat.pkl', 'rb'))
        wv_30commonnewsMedia_mat = pickle.load(open('pcl_files/wv_30commonnewsMedia_mat.pkl', 'rb'))

        logger.info("Loading pickle files done")

        app.logger.info('Predicting label...')
        labelProbLO_influencer=get_probability(processed_bio, wv_30commonInfluencer_mat, 
                                                                   followerCount_test, 
                                                                   coefficients_LO_Influencers)
    
        labelProbLO_brand=get_probability(processed_bio, wv_30commonBrand_mat, 
                                                         followerCount_test,coefficients_LO_brand)
    
        labelProbLO_newsMedia=get_probability(processed_bio, wv_30commonnewsMedia_mat, 
                                                                 followerCount_test,coefficients_LO_NewsMedia)

        predictions_array=np.concatenate((labelProbLO_influencer, labelProbLO_brand, labelProbLO_newsMedia),axis=0)
        labelIdx=np.argmax(predictions_array)
        predLabel=labels_dic.get(labelIdx)
        confidence=100/len(predictions_array[ np.where( predictions_array > 0 ) ])
        app.logger.info('Label prediction done')

        app.logger.info('Creating json output')           
        res={}
        res['predicted label'] = predLabel
        res['confidence'] = confidence

    else:
        res={}
        res['predicted label'] = 'maybe influencer'
        res['confidence'] = '30%'

        

    return json.dumps(res)

# To test use: curl -H "Content-Type: application/json" -X POST -d '{"name":"xyz","bio":"xyz", "follower_count": "1234"}' http://localhost:5000/echo
@app.route('/echo', methods=['GET', 'POST'])
def echo():
    if request.method == 'GET':
        app.logger.info('/echo hit via GET')
        return request.query_string
    else:
        app.logger.info('/echo hit via POST')
        jsonIn = request.get_json(force=True)
        return json.dumps(jsonIn)

if __name__ == "__main__":
    app.logger.info("Starting label prediction...")
    app.run()

