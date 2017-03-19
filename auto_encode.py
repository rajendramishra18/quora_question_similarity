from keras.models import Sequential
from keras.layers import Dense, Activation , Dropout
from keras.layers import LSTM
from keras.layers.noise import GaussianNoise
import cPickle as cp
import csv
from numpy.random import RandomState	
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import random
import collections
from itertools import groupby
from sets import Set
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from spacy.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from keras.optimizers import SGD , Adam
from sklearn import decomposition
from sklearn.decomposition import sparse_encode
from sklearn.metrics.pairwise import cosine_similarity



def parse_html(html_doc):
	clean_text = []
	for each in html_doc:
		soup = BeautifulSoup(each.decode("utf-8"), 'html.parser')
		clean_text.append(soup.get_text())
	return clean_text

def tokenize(doc , model):
	doc_list = []
	for each in doc:
		text = model(each.decode("utf_8"))
		pos_doc = []
		for sent in text.sents:
			for word in sent:
				tag = word.tag_
				pos = word.pos_
				#if 'NN' in tag or 'ADJ' in pos or 'ADV' in pos or 'VERB' in pos:
					#print(tag)
				pos_doc.append(word.lemma_.lower())
		doc_list.append(pos_doc)
	return doc_list


def remove_stopwords(stop, doc_list):
	cleaned_doc = []
	for doc in doc_list:
		temp = []
		for each in doc:
			if each not in stop:
				temp.append(each)
		cleaned_doc.append(temp)
						
	return cleaned_doc


def train_model(model , train_X):
	
	# We will store only the best possible model based on the val_loss. 
	# In turn it helps us avoiding overfitting.
	#checkpoint = ModelCheckpoint("model/weight_1.h5" ,  monitor='val_loss' , verbose = 1 , save_best_only=True)
	
	# EarlyStopping is again a measure to avoid overfitting.
	#es = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
	
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Compile the model with optimizer 'Adam' and loss 'binary_crossentropy'
	model.compile(loss='mse', optimizer='Adam')
	
	# save the model architecture
	##json_string = model.to_json()
	#fp = open("model/model_1_arch" , "w")
	#fp.write(json_string)
	
	# train the model for 100 epochs with earlystopping and batch size 32
	model.fit(train_X , train_X , batch_size = 64 , nb_epoch = 20 , verbose = 1 , validation_split = .2 , shuffle = True , )
	
	return model
	
def create_network():
	model = Sequential()
	model.add(GaussianNoise(0.01 , input_shape = (50,50)))
	
	# Encoder Architecture
	model.add(LSTM(5000 , return_sequences=True, stateful = False))
	model.add(Activation('relu'))
	
	
	# Decoder Architecture
	model.add(LSTM(50	 , return_sequences=True, stateful = False))
	model.add(Activation('relu'))
	
	return model



def prepare_chunk(sent_list):
	for i in range(0 , len(sent_list) , 100000):
		yield sent_list[i:i+100000]
		
		
def prepare_vectors(model , text):
	word_embed = []	
	count = 0
	for i in range(len(text)):
		sent = text[i]
		#print("sent",sent)
		word_vec = []
		for i in range(len(sent)):
			if sent[i] in model and i<30:
				##print sent[i]
				#print(model[each])
				#print("____________________________________________________________")
				word_vec+= list(model[sent[i]])
			if i>=30:
				count+=1
		
		#print(len(word_vec))
		word_embed.append(word_vec)
	print("No. of sents length greter than 30",count)
	return word_embed
	
	
def prepare_for_training(train_X):
	sent_mat = []
	for each in train_X:
		vec = np.zeros(1500)
		for i in range(len(each)):
			vec[i] = each[i]
		sent_mat.append(vec)
	return sent_mat

def train_sparse_coder():
		
	text = []
	topic = []
	tag = []
	
	list_doc = ["train.csv","test.csv"]
	p=3
	for i in range(len(list_doc)):
		with open(list_doc[i] , 'r') as datafile:
			data = csv.reader(datafile , delimiter = str(','))
			for row in data:
				#print(row)
				text.append(row[p])
				topic.append(row[p+1])
		p=1
			
	samp = random.sample(list(zip(topic, text)), 100000)
	cp.dump(samp , open("gen_sample.cp" , 'w'))
	topic , text = zip(*samp)
			
	# Extract raw text parsing the HTML
	#topic = parse_html(topic)
	
	#text = parse_html(text)
	
	text = text+topic
	len(text)
	
	# Perform tokenization and POS tagging
	model = English()
	sent_list = tokenize(text , model)
	
	
	# Remove stopwords
	#stop = set(stopwords.words('english'))
	#doc_cleaned = remove_stopwords(stop, sent_list)
	
	#cp.dump(doc_cleaned , open("exp.cp" , 'w'))
	
	'''
	# convert list into string for tfidf computation
	tfidf_matrix , feature_names = compute_tfidf(sent)
	'''
	word2vec = models.KeyedVectors.load_word2vec_format('../word_vector/model.txt', binary=False)
	'''
	chunk = prepare_chunk(doc_cleaned)
	model = create_network()
	print(model.summary())
	'''
	
	train_X = prepare_vectors(word2vec , sent_list)
	train_X = prepare_for_training(train_X)
	print("here")
	rng = RandomState(0)
	dict_estimator =decomposition.MiniBatchDictionaryLearning(n_components=100, alpha=0.1,n_iter=50, batch_size=32,random_state=rng)
	dict_estimator.fit(train_X)
	dictionary = dict_estimator.components_
	print dictionary
	np.save("dictionary.npy" , dictionary)
	print("computing code")
	#code = sparse_encode(train_X, dictionary)
	#np.save("code.npy" , dictionary)
	
	#print code[0]
	#print code.shape
	#print cosine_similarity(code[0] , code[1])	

	
def test_sparse_coder():
	text = []
	topic = []
	list_doc = ["train.csv"]
	for i in range(len(list_doc)):
		with open(list_doc[i] , 'r') as datafile:
			data = csv.reader(datafile , delimiter = str(','))
			for row in data:
				text.append(row[3])
				topic.append(row[4])
	
	print(text[0])

	samp = random.sample(list(zip(topic, text)), 10)
	topic , text = zip(*samp)
	#text = parse_html(text)
	
	# Perform tokenization and POS tagging
	model = English()
	text = tokenize(text , model)
	topic = tokenize(topic,model)
	sent_list = []
	for i in range(len(text)):
		sent_list.append(text[i])
		sent_list.append(topic[i])
	# Remove stopwords
	#stop = set(stopwords.words('english'))
	#doc_cleaned = remove_stopwords(stop, sent_list)
	
	#cp.dump(doc_cleaned , open("exp.cp" , 'w'))
	
	'''
	# convert list into string for tfidf computation
	tfidf_matrix , feature_names = compute_tfidf(sent)
	'''
	word2vec = models.KeyedVectors.load_word2vec_format('../word_vector/model.txt', binary=False)
	'''
	chunk = prepare_chunk(doc_cleaned)
	model = create_network()
	print(model.summary())
	'''
	
	test_X = prepare_vectors(word2vec , sent_list)
	test_X = np.array(prepare_for_training(test_X))
	print(test_X.shape)
	#rng = RandomState(0)
	#dict_estimator =decomposition.MiniBatchDictionaryLearning(n_components=100, alpha=0.1,n_iter=50, batch_size=32,random_state=rng)
	#dict_estimator.fit(train_X)
	#dictionary = dict_estimator.components_
	#print dictionary
	#np.save("dictionary.npy" , dictionary)
	dictionary = np.load("dictionary.npy")
	code = sparse_encode(test_X, dictionary)
	i=0
	while i<len(test_X):
		print("_________________________________________________________________________________________")
		print sent_list[i]
		print("*******************************************************************************************")
		print sent_list[i+1]
		print cosine_similarity(code[i] , code[i+1])
		print(code[i]-code[i+1])
		i+=2	

#train_sparse_coder()
test_sparse_coder()
	
