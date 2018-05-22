import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import gensim, logging
import re
import string
from collections import Counter
import math

stemmer = SnowballStemmer("english")
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
stopwords = nltk.corpus.stopwords.words('english')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def remove_stop_words(transcipt_list,stopwords):
    trans_list_cleaned=list()
    for word in transcipt_list:
        if word.lower() not in stopwords:
            trans_list_cleaned.append(word)
    return trans_list_cleaned

def stemming(text_list, stemmer):
    stemmed_list=list()
    for lst in text_list:
        sentence=list()
        for word in lst:
            stemmed_list.append(stemmer.stem(word.lower()))
    return stemmed_list

def remove_regexp(bio):
    rebio=re.sub(r'[^\w]',' ',bio)
    return rebio

def stem_tokenize_stopWords(trainLabel):
    tokenizedBios=list()
    tokenizedBiosWords=list()
    bio=trainLabel['bio']
    rebio=remove_regexp(bio)
    sentences_list=tokenizer.tokenize(rebio)
    tokenized_text=[sentence.split() for sentence in sentences_list]
    tokinezed_text_stemmed=stemming(tokenized_text, stemmer)
    tokinezed_text_stemmed_nostopwords=remove_stop_words(tokinezed_text_stemmed,stopwords)
    tokenizedBios.append(tokinezed_text_stemmed_nostopwords)
    tokenizedBiosWords=tokenizedBiosWords+tokinezed_text_stemmed_nostopwords
    return tokenizedBios, tokenizedBiosWords

def counters(tokinezedSentences, tokenizedWords):
    count_list=list()

    for sentence in tokinezedSentences:
        count_list.append(Counter(sentence))

    word_counts=Counter(tokenizedWords)

    return word_counts, count_list

def vector_representation_mat(tokenized_text, word_counts, count_list, model, dimension):

    Vik_vec_norm_idf=np.zeros((len(tokenized_text),dimension))

    for s,sentence in enumerate(tokenized_text):

            Vik_wordnorm_vecnorm_sum=np.zeros(dimension)
            keys=list(count_list[s].keys())
            values=list(count_list[s].values())
            for v, key in enumerate(keys):
                if key in word_counts.keys():
                    word_norm=math.log(len(tokenized_text)/word_counts[key])
                else:
                    word_norm=1
                if key in model:
                    word_vec=model[key]
                    vec_norm=np.sum(np.power(word_vec,2))
                else:
                    word_vec=np.zeros((dimension,1))  #won't contribute to sum
                    vec_norm=1
                Vik_wordnorm_vecnorm_sum+=(word_vec.flat/np.sqrt(vec_norm))*float(values[v])*word_norm
            Vik_vec_norm_idf[s,:]=Vik_wordnorm_vecnorm_sum

    return Vik_vec_norm_idf

def create_wvMat(pdData, model, dimension):
    tokenizedBios, tokenizedBiosWords=stem_tokenize_stopWords(pdData)
    wordCount, listCounts = counters(tokenizedBios, tokenizedBiosWords)
    wvMat=vector_representation_mat(tokenizedBios, wordCount, listCounts, model, dimension)
    return wvMat
