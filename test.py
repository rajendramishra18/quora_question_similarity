from sklearn import svm
from auto_encode import tokenize , prepare_vectors , prepare_for_training , sparse_encode , prepare_chunk
import cPickle as cp
import csv
import numpy as np
from numpy.random import RandomState	
from sklearn.cluster import MiniBatchKMeans
from spacy.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from sklearn.decomposition import sparse_encode

def prepare_train_data():
	text1 = []
	text2 = []
	qid = []
	list_doc = ["test.csv"]
	for i in range(len(list_doc)):
		with open(list_doc[i] , 'r') as datafile:
			data = csv.reader(datafile , delimiter = str(','))
			for row in data:
				text1.append(row[1])
				text2.append(row[2])
				qid.append(row[0])
	
	#samp = random.sample(list(zip(topic, text)), 10)
	#topic , text = zip(*samp)
	#text = parse_html(text)
	print(len(text1))
	print(len(text2))
	print(len(qid))

	text1 = list(prepare_chunk(text1))
	text2 = list(prepare_chunk(text2))
	qid = list(prepare_chunk(qid))
	# Perform tokenization and POS tagging
	model = English()
	word2vec = models.KeyedVectors.load_word2vec_format('../word_vector/model.txt', binary=False)
	dictionary = np.load("dictionary.npy")
	p=0
	for each in text1:
		print("chunk number" , p)
		chunk1 = tokenize(text1[p] , model)
		chunk2 = tokenize(text2[p],model)
		chunkq = qid[p]
		#print(chunk_tag[0])
		# Remove stopwords
		#stop = set(stopwords.words('english'))
		#doc_cleaned = remove_stopwords(stop, sent_list)
	
		#cp.dump(doc_cleaned , open("exp.cp" , 'w'))
		
		'''
		# convert list into string for tfidf computation
		tfidf_matrix , feature_names = compute_tfidf(sent)
		word2vec = models.KeyedVectors.load_word2vec_format('../word_vector/model.txt', binary=False)

		chunk = prepare_chunk(doc_cleaned)
		model = create_network()
		print(model.summary())
		'''
	
		#test_X = prepare_vectors(word2vec , sent_list)
		chunk1 = prepare_vectors(word2vec , chunk1)
		chunk2 = prepare_vectors(word2vec , chunk2)
		#test_X = np.array(prepare_for_training(test_X))
		chunk1 = np.array(prepare_for_training(chunk1))
		chunk2 = np.array(prepare_for_training(chunk2))
		#rng = RandomState(0)
		#dict_estimator =decomposition.MiniBatchDictionaryLearning(n_components=100, alpha=0.1,n_iter=50, batch_size=32,random_state=rng)
		#dict_estimator.fit(train_X)
		#dictionary = dict_estimator.components_
		#print dictionary
		#np.save("dictionary.npy" , dictionary)
		#dictionary = np.load("dictionary.npy")
		print("computing code")
		code1 = sparse_encode(chunk1, dictionary)
		code2 = sparse_encode(chunk2 , dictionary)
		print("done computing code")
		test_X = []
		i=0
		while i<len(chunk1):
			#print("_________________________________________________________________________________________")
			#print sent_list[i]
			#print("*******************************************************************************************")
			#print sent_list[i+1]
			#print cosine_similarity(code[i] , code[i+1])
			test_X.append(code1[i]-code2[i])
			i+=1
		
		#fname_X = "train_X_"+str(p)
		fmane_X = "test_X_"+str(p)
		#cp.dump(train_X , open(fname_X , 'w'))
		cp.dump(test_X , open(fmane_X	 , 'w'))
		fnameid = "id"+str(p)
		cp.dump(chunkq , open(fnameid,'w'))
		p+=1
prepare_train_data()	
