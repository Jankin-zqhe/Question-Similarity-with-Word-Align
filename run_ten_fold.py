#coding=utf-8
import sys
import os
import pandas as pd
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


def partition(v,gpu,prefix='../data/',file='char_train'):

	fr = open(os.path.join(prefix,file),'r')
	tuples = fr.readlines()
	fr.close()
	
	lens = len(tuples)
	order = range(lens)
	np.random.shuffle(order)


	fold = 10
	fold_size = lens/10+1
	
	for j in range(fold):
		dev_list = order[j*fold_size:(j+1)*fold_size]
		train_list = order[:j*fold_size] + order[(j+1)*fold_size:]


		ftrain = open(os.path.join(prefix,'train_fold.%s.%d.txt' % (v,j)),'w')
		fdev = open(os.path.join(prefix,'dev_fold.%s.%d.txt' % (v,j)),'w')
	

		for i in train_list:
			ftrain.write(tuples[i].strip()+'\n')
		

		for i in dev_list:
			fdev.write(tuples[i].strip()+'\n')
			

		ftrain.close()
		fdev.close()
		comand = 'python -u ../code/ten_fold.py --gpu %s --batch 128 --drop 0.5 --v %s.%d >log.v%s.%d 2>log.err.v%s.%d' % (gpu,v,j,v,j,v,j)
		os.system(comand)

partition(sys.argv[1],sys.argv[2])