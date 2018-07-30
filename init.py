#coding=utf-8
import os
import numpy as np
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def getEmbed(prefix='../data/',file='word2vec.txt'):
# def getEmbed(prefix='../data/',file='sgns.zhihu.word.ccks'):
	fr = open(os.path.join(prefix,file),'r')
	wordVec = []
	wordMap = {}

	fr.readline()
	while True:
		line = fr.readline()
		if not line:
			break
		content = map(float,line.strip().split()[1:])
		w = line.strip().split()[0]

		wordMap[w] = len(wordVec)
		wordVec.append(content)

	wordMap['UNK'] = len(wordMap)
	wordMap['BLANK'] = len(wordMap)
	wordVec.append([0.0]*300)
	wordVec.append([0.0]*300)
	wordVec = np.reshape(np.array(wordVec,dtype=np.float32),(-1,300))

	return wordMap,wordVec
def getEmbed2(prefix='../data/',file='train_set.txt'):
	fr = open(os.path.join(prefix,file),'r')

	wordMap = {}
	wordNum = {}
	while True:
		line = fr.readline()
		if not line:
			break
		words = line.strip().split()[:-1]
		for w in words:
			if w in wordNum:
				wordNum[w]+=1
			else:
				wordNum[w] =1
	wordMap['BLANK'] = 0
	wordMap['UNK'] = 1
	count = 2
	for w in wordNum:
		if wordNum[w]>5:
			wordMap[w] = count
			count+=1
	fr.close()
	return wordMap

def partition(prefix='../data',file='train_seg.txt'):
	fr = open(os.path.join(prefix,file),'r')
	lines = fr.readlines()
	lens = len(lines)
	order = range(lens)
	np.random.shuffle(order)


	train_list = order[:int(lens*0.8)]
	dev_list = order[int(lens*0.8):int(lens*0.9)]
	test_list = order[int(lens*0.9):]

	ftrain = open(os.path.join(prefix,'train_set.txt'),'w')
	fdev = open(os.path.join(prefix,'dev_set.txt'),'w')
	ftest = open(os.path.join(prefix,'test_set.txt'),'w')
	for i in train_list:
		ftrain.write(lines[i].strip()+'\n')
	for i in dev_list:
		fdev.write(lines[i].strip()+'\n')
	for i in test_list:
		ftest.write(lines[i].strip()+'\n')

	fr.close()
	ftrain.close()
	fdev.close()
	ftest.close()
# partition()

def stopWords(file='../data/stop_words.txt'):
	fr = open(file,'r')
	stop_words = {}
	while True:
		line = fr.readline()
		if not line:
			break
		word = line.strip().split()[0]
		stop_words[word] = 1
	fr.close()
	return stop_words


def getQuestion(wordMap,prefix='../data/',file='train_set.txt',word_size=30): #word_size 41 , char_size 60
	fr = open(os.path.join(prefix,file),'r')

	
	train_set = []

	stop_words = stopWords()
	while True:
		line = fr.readline()
		if not line:
			break
		ques = line.strip().split('\t')
		label = int(ques[0])
		q1 = ques[1].split()
		q2 = ques[2].split()
		# q1 = q1#[:word_size]
		words1 = []
		for w in q1:
			if len(words1)>=word_size:
				break
			if w in stop_words:
				continue
			if w in wordMap:
				w = wordMap[w]
			else:
				w = wordMap['UNK']
			words1.append(w)
		if len(words1) == 0:
			words1.append(wordMap['UNK'])
		mask = [0.0]*min(len(words1),word_size)
		mask01 = [1.0]*min(len(words1),word_size)
		while len(words1)<word_size:
			words1.append(wordMap['BLANK'])
			mask.append(-10000000.0)
			mask01.append(0.0)


		# q2 = q2#[:word_size]
		words2 = []
		for w in q2:
			if len(words2)>=word_size:
				break
			if w in stop_words:
				continue
			if w in wordMap:
				w = wordMap[w]
			else:
				w = wordMap['UNK']
			words2.append(w)
		if len(words2) == 0:
			words2.append(wordMap['UNK'])

		mask2 = [0.0]*min(len(words2),word_size)
		mask012 = [1.0]*min(len(words2),word_size)
		while len(words2)<word_size:
			words2.append(wordMap['BLANK'])
			mask2.append(-10000000.0)
			mask012.append(0.0)

		train_set.append((words1,mask,mask01,words2,mask2,mask012,label))

	fr.close()
	return train_set

def getQuestionTest(wordMap,prefix='../data/',file='dev_seg.txt',word_size=30): #word_size 41 , char_size 60
	fr = open(os.path.join(prefix,file),'r')

	
	train_set = []
	stop_words = stopWords()

	while True:
		line = fr.readline()
		if not line:
			break
		ques = line.strip().split('\t')
		# label = int(ques[-1])
		q1 = ques[1].split()
		q2 = ques[2].split()
		# q1 = q1#[:word_size]
		words1 = []
		for w in q1:
			if len(words1)>=word_size:
				break
			if w in stop_words:
				continue
			if w in wordMap:
				w = wordMap[w]
			else:
				w = wordMap['UNK']
			words1.append(w)
		if len(words1) == 0:
			words1.append(wordMap['UNK'])
		mask = [0.0]*min(len(words1),word_size)
		mask01 = [1.0]*min(len(words1),word_size)
		while len(words1)<word_size:
			words1.append(wordMap['BLANK'])
			mask.append(-10000000.0)
			mask01.append(0.0)


		# q2 = q2#[:word_size]
		words2 = []
		for w in q2:
			if len(words2)>=word_size:
				break
			if w in stop_words:
				continue
			if w in wordMap:
				w = wordMap[w]
			else:
				w = wordMap['UNK']
			words2.append(w)
		if len(words2) == 0:
			words2.append(wordMap['UNK'])
		mask2 = [0.0]*min(len(words2),word_size)
		mask012 = [1.0]*min(len(words2),word_size)
		while len(words2)<word_size:
			words2.append(wordMap['BLANK'])
			mask2.append(-10000000.0)
			mask012.append(0.0)

		train_set.append((words1,mask,mask01,words2,mask2,mask012))

	fr.close()
	return train_set



