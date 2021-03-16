import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk import WordNetLemmatizer
from PyPDF2 import PdfFileWriter




'''with open('test result.txt', 'r') as file:
        doc = file.read()

doc = doc.replace('.', '. ')
tokenize_sent = sent_tokenize(doc)'''


def regex(doc):
    regex = r'[^a-zA-Z0-9\s]'
    text_ = [re.sub(regex, '', i) for i in doc]
    return text_



def text_preprocessing(text):
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    new_tokens = [i for i in tokens if i not in stopwords.words('english')]
    tokens = [word for word in new_tokens if len(word) >= 3]
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(word) for word in tokens]
    
    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    lemmatizer = WordNetLemmatizer()
    
    def prac_lemmatize(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token, 'n')
    lemmatized_text = ' '.join([prac_lemmatize(token, tag) for token, tag in tagged_corpus])
    
    return lemmatized_text 


def summarize(text_):

	final_text = np.array([text_preprocessing(i) for i in text_])
	vectorizer = TfidfVectorizer(strip_accents = 'unicode', norm = 'l2')
	matrix = vectorizer.fit_transform(final_text).todense()


	sent_score = matrix.sum(axis=1)
	sent_score_total = sent_score.sum()
	average = sent_score_total/sent_score.shape[0]

	summary = []

	for i in range(sent_score.shape[0]):
	    if sent_score[i] >= average:
	        summary.append(tokenize_sent[i])

	sum3 = ''.join(summary)
	return sum3
