import tensorflow as tf
import numpy as np
import time
import datetime
import os
from init import *
import network
import pandas as pd
import random


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary_dir','../summary/','path to store summary')
tf.app.flags.DEFINE_string('gpu','','gpu id')
tf.app.flags.DEFINE_string('v','1','model version')
tf.app.flags.DEFINE_string('batch','128','batch size')
tf.app.flags.DEFINE_string('epochs','22','train epochs')
tf.app.flags.DEFINE_string('dim','300','dimension')
tf.app.flags.DEFINE_string('target','../model','save model dir')
tf.app.flags.DEFINE_string('source','../data','data dir to load')
tf.app.flags.DEFINE_string('train','1','is train')
tf.app.flags.DEFINE_string('id','14000','model id')
# tf.app.flags.DEFINE_string('word','25','word len')
tf.app.flags.DEFINE_string('word','30','word len')

tf.app.flags.DEFINE_string('char','35','char len')
tf.app.flags.DEFINE_string('drop','0.5','drop')
tf.app.flags.DEFINE_string('lr','0.001','drop')
def loadData(prefix,file):
	fr = open(os.path.join(prefix,file),'r')
	train_list = []
	while True:
		line = fr.readline()
		if not line:
			break
		train_list.append(map(int,line.strip().split()))
	fr.close()
	return train_list


def main(_):
	
	hidden_dim = int(FLAGS.dim)
	batch = int(FLAGS.batch)

	is_train = int(FLAGS.train)
	source = FLAGS.source
	target = FLAGS.target
	total_epochs = int(FLAGS.epochs)
	word_len = int(FLAGS.word)
	char_len = int(FLAGS.char)
	drop = float(FLAGS.drop)

	wordMap,wordVec = getEmbed()
	dict_size = len(wordMap)
	
	

	train_set = getQuestion(wordMap,source,'train_fold.%s.txt' % FLAGS.v)
	
	dev_set = getQuestion(wordMap,source,'dev_fold.%s.txt' % FLAGS.v)


	test_set = getQuestionTest(wordMap,source,'ccks_test')
	# test_set = getQuestionTest(wordMap,source,'char_dev')

	

	gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu,allow_growth=True)

	with tf.Graph().as_default():
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
		with sess.as_default():

			
			initializer = tf.contrib.layers.xavier_initializer()
			# initializer = tf.orthogonal_initializer()
			with tf.variable_scope('model',initializer=initializer):
				if FLAGS.v.split('.')[0] == 'lstm_lstm_att':
					m = network.Question_Similarity_with_Word_Align(wordVec,word_len,dict_size=dict_size)
				elif FLAGS.v.split('.')[0] == 'lstm_with_other':
					m = network.Question_Similarity_with_Word_Align(wordVec,word_len,dict_size=dict_size)


			global_step = tf.Variable(0,name='global_step',trainable=False)
			# optimizer = tf.train.AdamOptimizer(float(FLAGS.lr))
			# optimizer = tf.train.RMSPropOptimizer(float(FLAGS.lr))

			lr = tf.train.exponential_decay(float(FLAGS.lr),global_step=global_step,decay_steps=500,decay_rate=0.98)
			optimizer = tf.train.AdamOptimizer(lr)

			# optimizer = tf.train.MomentumOptimizer(lr,momentum=0.9)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(m.loss,global_step=global_step)
			
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver(max_to_keep=None)
				# merged_summary = tf.summary.merge_all()
				# summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)
			
			def train_step(q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01):
				feed_dict = {}

				feed_dict[m.q1_word] = q1_word
				feed_dict[m.q1_word_mask] = q1_word_mask
		

				feed_dict[m.q1_word_mask01] = q1_word_mask01
				feed_dict[m.q2_word_mask01] = q2_word_mask01

				feed_dict[m.q2_word] = q2_word
				feed_dict[m.q2_word_mask] = q2_word_mask
	
				feed_dict[m.is_training]=True

				feed_dict[m.label] = labels

				feed_dict[m.keep_prob] = drop


				_,step,loss,accuracy  = sess.run([train_op, global_step, m.test_loss, m.accuracy],feed_dict)
				time_str = datetime.datetime.now().isoformat()
				# accuracy = np.reshape(np.array(accuracy),(batch))
				# acc = np.mean(accuracy)
				# summary_writer.add_summary(summary,step)

				if step % 10 == 0:
					tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, np.mean(loss), accuracy)
					print tempstr

			def dev_step(q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01):
				feed_dict = {}

				feed_dict[m.q1_word] = q1_word
				feed_dict[m.q1_word_mask] = q1_word_mask


				feed_dict[m.q1_word_mask01] = q1_word_mask01
				feed_dict[m.q2_word_mask01] = q2_word_mask01

				feed_dict[m.q2_word] = q2_word
				feed_dict[m.q2_word_mask] = q2_word_mask

				feed_dict[m.is_training]=False

				feed_dict[m.label] = labels

				feed_dict[m.keep_prob] = 1.0
				loss, accuracy,prediction = sess.run([m.test_loss,m.test_accuracy,m.prediction],feed_dict)

				return loss,accuracy,prediction

			def test_step(q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01):
				feed_dict = {}

				feed_dict[m.q1_word] = q1_word
				feed_dict[m.q1_word_mask] = q1_word_mask
		

				feed_dict[m.q2_word] = q2_word
				feed_dict[m.q2_word_mask] = q2_word_mask
			
				feed_dict[m.is_training]=False
				feed_dict[m.q1_word_mask01] = q1_word_mask01
				feed_dict[m.q2_word_mask01] = q2_word_mask01

				feed_dict[m.keep_prob] = 1.0
				prob,prediction = sess.run([m.prob,m.prediction],feed_dict)
				return prob,prediction

			def getData(lst,dataset,reverse=False):
				q1_word = []
				q1_word_mask = []


				q1_word_mask01 = []

				q2_word = []
				q2_word_mask = []


				q2_word_mask01 = []

				labels = []
				all_labels = []

				for k in lst:
					words1,mask,mask01,words2,mask2,mask012,l = dataset[k]
					if l == 1:
						labels.append([0,1])
					elif l==0:
						labels.append([1,0])
					else:
						print('error')
					all_labels.append(l)
					# rand = random.randint(0,1)
					# if rand==1:
					# if reverse==True:
					# 	q1,q2 = q2,q1

					q1_word.append(words1)
					q1_word_mask.append(mask)

					q1_word_mask01.append(mask01)

					q2_word.append(words2)
					q2_word_mask.append(mask2)


					q2_word_mask01.append(mask012)

				q1_word = np.array(q1_word)
				q1_word_mask = np.array(q1_word_mask)
	

				q1_word_mask01 = np.array(q1_word_mask01)

				q2_word = np.array(q2_word)
				q2_word_mask = np.array(q2_word_mask)

				q2_word_mask01 = np.array(q2_word_mask01)

				labels = np.array(labels)

				return q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01,all_labels

			def evaluate(total_labels,total_pred):
				a = np.sum(total_labels)
				b = np.sum(total_pred)
				c = 0
				for i,j in zip(total_labels,total_pred):
					if i==1 and j==1:
						c+=1
				if b<=0:
					precision=0.0
				else:
					precision = float(c)/float(b)
				recall = float(c)/float(a)
				f1 = 2*precision*recall/(precision+recall)
				return precision,recall,f1


			
			max_accuracy = 0.0
			min_loss = 100000000
			max_f1 = 0.0
			for one_epoch in range(total_epochs):
				reverse = False
				if one_epoch % 2 == 1:
					reverse = True
				print('turn: ' + str(one_epoch))
				temp_order = range(len(train_set))
				np.random.shuffle(temp_order)
				for i in range(int(len(temp_order)/float(batch))):
					
					temp_input = temp_order[i*batch:(i+1)*batch]


					q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01,all_l = getData(temp_input,train_set,reverse=reverse)
					train_step(q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01)

					current_step = tf.train.global_step(sess,global_step)
					if (current_step%150==0):
						accuracy = []
						losses=[]
						total_pred=[]
						total_labels = []
						dev_order = range(len(dev_set))
						for i in range(int(len(dev_order)/float(batch))):
							temp_input = dev_order[i*batch:(i+1)*batch]
							q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01,all_l = getData(temp_input,dev_set)
							loss,test_accuracy,prediction = dev_step(q1_word,q1_word_mask,q2_word,q2_word_mask,labels,q1_word_mask01,q2_word_mask01)
					#if current_step == 50:
							for acc in test_accuracy:
								accuracy.append(acc)
							for los in loss:
								losses.append(los)
							for p in prediction:
								total_pred.append(p)
							for l in all_l:
								total_labels.append(l)
			
						total_pred = np.reshape(np.array(total_pred),(-1)).tolist()
						precision,recall,f1 = evaluate(total_labels,total_pred)

						accuracy = np.reshape(np.array(accuracy),(-1))
						accuracy = np.mean(accuracy)
						losses = np.reshape(np.array(losses),(-1))
						losses = np.mean(losses)
						print('dev...')

						if f1 > max_f1:
							# if accuracy > max_accuracy:
							# max_accuracy = accuracy
							max_f1 = f1
							
							# if losses < min_loss:
							# 	min_loss = losses
							print('accuracy:  ' + str(accuracy))
							print('precision: ' + str(precision))
							print('recall:    ' + str(recall))
							print('f1:        ' + str(f1)) 
							# print('loss: ' + str(losses))

							# if accuracy < 91 and min_loss>0.2:
							# # if min_loss>0.2:
							# 	continue

							print 'saving model'
							# path = saver.save(sess,target +'/CNN_model.'+FLAGS.v,global_step=current_step)
							path = saver.save(sess,target +'/ccks_model.'+FLAGS.v,global_step=0)
							tempstr = 'have saved model to '+path
							print tempstr
			def getDataT(lst,dataset):
				q1_word = []
				q1_word_mask = []
				q1_char = []
				q1_char_mask = []

				q2_word = []
				q2_word_mask = []
				q2_char = []
				q2_char_mask = []

				q1_word_mask01 = []

				q2_word_mask01 = []


				labels = []

				for k in lst:
					# q1, q2 = dataset[k]
					words1,mask,mask01,words2,mask2,mask012 = dataset[k]

					q1_word.append(words1)
					q1_word_mask.append(mask)


					q2_word.append(words2)
					q2_word_mask.append(mask2)
	
					q1_word_mask01.append(mask01)

					q2_word_mask01.append(mask012)


				q1_word = np.array(q1_word)
				q1_word_mask = np.array(q1_word_mask)


				q2_word = np.array(q2_word)
				q2_word_mask = np.array(q2_word_mask)


				q1_word_mask01 = np.array(q1_word_mask01)

				q2_word_mask01 = np.array(q2_word_mask01)
				


				return q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01

			def make_submission(pred):
				# import numpy as np
				# fr = open(os.path.join(source,'dev.txt'),'r')
			
				
					
				ids = range(110000)
				pred = pred.tolist()
				data = {'test_id':ids,'result':pred}
				dataframe = pd.DataFrame(data,columns=['test_id','result'])
				dataframe.to_csv(os.path.join(source,'result'+ FLAGS.v+'.csv'),index=False,sep=',')

			path = target +'/ccks_model.'+FLAGS.v+'-'+'0'
			# path = target +'/CNN_model.'+FLAGS.v+'-'+FLAGS.id	
			print 'load model:',path
			saver = tf.train.Saver()
			saver.restore(sess,path)
			print 'end load model'

			total_prob = []
			total_pred = []
			temp_order = range(len(test_set))
			for i in range(0,len(test_set),batch):
				temp_input = range(i,min(len(test_set),i+batch))
				q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01 = getDataT(temp_input,test_set)
				prob,prediction = test_step(q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01)
		#if current_step == 50:
				prob = np.reshape(prob,(-1,2))
				for p in prob:
					total_prob.append(p)
				for p in prediction:
					total_pred.append(p)
			total_prob = np.reshape(np.array(total_prob),(-1,2))
			total_pred = np.reshape(np.array(total_pred),(-1))
			
			make_submission(total_pred)







	                        


	        # else:




if __name__ == '__main__':
	tf.app.run()
