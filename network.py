import tensorflow as tf
import numpy as np
import sys

import tensorflow as tf
import numpy as np
import sys



class Question_Similarity_with_Word_Align: # cnn--> lstm --> cnnatt
	def __init__(self,wordVec,word_len,dict_size):
	# def __init__(self,word_len,dict_size):
		self.batch = 50
		self.window = 3

		self.windows=[2,3,4]

		self.word_len = word_len

		self.input_dim = 300
		self.hidden_dim = 600

		self.q1_word = tf.placeholder(dtype=tf.int32,shape=[None,self.word_len],name='q1_word')
		
		# self.q1_word_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len - self.window +1],name='q1_word_mask')


		self.q1_word_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len],name='q1_word_mask')
		self.q1_word_mask01 = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len],name='q1_word_mask01')

		# self.q1_char_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.char_len],name='q1_char_mask')

		self.q2_word = tf.placeholder(dtype=tf.int32,shape=[None,self.word_len],name='q2_word')
		


		self.q2_word_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len],name='q2_word_mask')
		self.q2_word_mask01 = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len],name='q2_word_mask01')

		# self.q2_char_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.char_len],name='q2_char_mask')
		# self.q2_word_mask = tf.placeholder(dtype=tf.float32,shape=[None,self.word_len - self.window +1],name='q2_word_mask')
		

		self.label = tf.placeholder(dtype=tf.int32,shape=[None,2],name='label')

		self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
		self.is_training = tf.placeholder(dtype=tf.bool,name='is_training')

		# self.word_embedding = tf.get_variable(name='word_embedding', shape=[dict_size,self.input_dim])
		self.word_embedding = tf.get_variable(name='word_embedding', initializer=wordVec)
		
		# self.char_embedding = tf.concat(axis=0,values=[self.char_embedding,tmp])


		self.word_att_w = tf.get_variable(name='word_att_w',shape=[3*self.hidden_dim,self.hidden_dim])
		self.word_att_b = tf.get_variable(name='word_att_b',shape=[self.hidden_dim])
		self.word_att_v = tf.get_variable(name='word_att_v',shape=[self.hidden_dim,1])


		self.word_att = tf.get_variable(name='word_att',shape=[self.hidden_dim,1])
		self.word_att2 = tf.get_variable(name='word_att2',shape=[self.hidden_dim,1])
		self.char_att = tf.get_variable(name='char_att',shape=[self.hidden_dim,1])

		self.char_att_w = tf.get_variable(name='char_att_w',shape=[2*self.hidden_dim,self.hidden_dim])
		self.char_att_b = tf.get_variable(name='char_att_b',shape=[self.hidden_dim])
		self.char_att_v = tf.get_variable(name='char_att_v',shape=[self.hidden_dim,1])

		


		# self.char_w = tf.get_variable(name='char_w',shape=[self.window,self.input_dim,1,self.hidden_dim])
		# self.char_b = tf.get_variable(name='char_b',shape=[self.hidden_dim])
		
		# with tf.variable_scope('RNN', initializer=tf.orthogonal_initializer()):
		with tf.variable_scope('RNN'):# initializer=tf.orthogonal_initializer()):

			layer1_forward = self.LSTM()
			layer1_backward = self.LSTM()

			layer2_forward = self.LSTM()
			layer2_backward = self.LSTM()

			# layer3_forward = self.LSTM()
			# layer3_backward = self.LSTM()

			# layer4_forward = self.LSTM()
			# layer4_backward = self.LSTM()

		
		self.q1_word_all1,self.q1_word_rep1 = self.embed_l1(layer1_forward,layer1_backward,self.q1_word,self.word_embedding,self.word_att,self.word_len,self.q1_word_mask,self.q1_word_mask01,scope='WORD_l1')
		self.q2_word_all1,self.q2_word_rep1 = self.embed_l1(layer1_forward,layer1_backward,self.q2_word,self.word_embedding,self.word_att,self.word_len,self.q2_word_mask,self.q2_word_mask01,scope='WORD_l1',reuse=True)
		
		

		zero_mat1 = tf.constant(0.0,shape=[1,1,self.word_len,self.hidden_dim])
		zero_mat2 = tf.constant(0.0,shape=[1,self.word_len,1,self.hidden_dim])

		self.q1_word_simi = tf.reshape(self.q1_word_all1,[-1,self.word_len,1,self.hidden_dim]) - zero_mat1
		self.q2_word_simi = tf.reshape(self.q2_word_all1,[-1,1,self.word_len,self.hidden_dim]) - zero_mat2

		simi_mat = tf.reshape(tf.concat(axis=3,values=[self.q1_word_simi,self.q2_word_simi]),[-1,2*self.hidden_dim])
		scores = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(simi_mat,self.char_att_w) + self.char_att_b),self.char_att_v),[-1,self.word_len,self.word_len])
		q1_simi = tf.nn.softmax(scores,2)

		simi_mat = tf.reshape(tf.concat(axis=3,values=[self.q2_word_simi,self.q1_word_simi]),[-1,2*self.hidden_dim])
		scores = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(simi_mat,self.char_att_w) + self.char_att_b),self.char_att_v),[-1,self.word_len,self.word_len])
		q2_simi = tf.nn.softmax(tf.transpose(scores,[0,2,1]),2)

		self.q1_pad_q2 = tf.reshape(tf.matmul(q1_simi,self.q2_word_all1),[-1,self.word_len,self.hidden_dim])
		self.q2_pad_q1 = tf.reshape(tf.matmul(q2_simi,self.q1_word_all1),[-1,self.word_len,self.hidden_dim])

		# q1_simi_neg = tf.nn.softmax(tf.negative(scores),1)
		# q2_simi_neg = tf.nn.softmax(tf.transpose(tf.negative(scores),[0,2,1]),1)

		# self.q1_pad_q2_neg = tf.reshape(tf.matmul(q1_simi_neg,self.q2_word_all1),[-1,self.word_len,self.hidden_dim])
		# self.q2_pad_q1_neg = tf.reshape(tf.matmul(q2_simi_neg,self.q1_word_all1),[-1,self.word_len,self.hidden_dim])

		# self.q1_word_all2 = tf.concat(axis=2,values=[self.q1_word_all1,self.q1_pad_q2,self.q1_word_all1-self.q1_pad_q2,tf.multiply(self.q1_word_all1,self.q1_pad_q2),self.q1_pad_q2_neg])
		# self.q2_word_all2 = tf.concat(axis=2,values=[self.q2_word_all1,self.q2_pad_q1,self.q2_word_all1-self.q2_pad_q1,tf.multiply(self.q2_word_all1,self.q2_pad_q1),self.q2_pad_q1_neg])

		self.q1_word_all2 = tf.concat(axis=2,values=[self.q1_word_all1,self.q1_pad_q2,self.q1_word_all1-self.q1_pad_q2,tf.multiply(self.q1_word_all1,self.q1_pad_q2)])
		self.q2_word_all2 = tf.concat(axis=2,values=[self.q2_word_all1,self.q2_pad_q1,self.q2_word_all1-self.q2_pad_q1,tf.multiply(self.q2_word_all1,self.q2_pad_q1)])

		# self.q1_word_all2 = tf.layers.batch_normalization(self.q1_word_all2,training=self.is_training)
		# self.q2_word_all2 = tf.layers.batch_normalization(self.q2_word_all2,training=self.is_training)
		# self.q1_word_all1 = tf.layers.batch_normalization(self.q1_word_all1,training=self.is_training)
		
		_,self.q1_word_rep2 = self.embed_l1(layer2_forward,layer2_backward,self.q1_word_all2,self.word_embedding,self.word_att2,self.word_len,self.q1_word_mask,self.q1_word_mask01,scope='WORD_l2',raw=False)
		_,self.q2_word_rep2 = self.embed_l1(layer2_forward,layer2_backward,self.q2_word_all2,self.word_embedding,self.word_att2,self.word_len,self.q2_word_mask,self.q2_word_mask01,scope='WORD_l2',reuse=True,raw=False)

		# self.q1_word_rep,self.q1_word_rep2 = self.embed(self.q2_word_rep1,word_forward,word_backward,self.q1_word_all,self.word_att_w,self.word_att_b,self.word_att_v,self.word_len,self.q1_word_mask,self.q1_word_mask01,scope='WORD')

		# self.q2_word_rep,self.q2_word_rep2 = self.embed(self.q1_word_rep1,word_forward,word_backward,self.q2_word_all,self.word_att_w,self.word_att_b,self.word_att_v,self.word_len,self.q2_word_mask,self.q2_word_mask01,scope='WORD',reuse=True)
		q1_word_mask

		self.deal6()
		# self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.00001),weights_list=tf.trainable_variables())
		# self.loss += self.l2_loss #tf.cast(self.l2_loss,tf.float64)
	
		# self.deal7()
		# self.deal8()
		# self.deal9()
		# self.deal10()


	def deal6(self):

		

		self.q1 = tf.concat(axis=1,values=[self.q1_word_rep1,self.q1_word_rep2])
		self.q2 = tf.concat(axis=1,values=[self.q2_word_rep1,self.q2_word_rep2])

		# self.q1 = tf.layers.batch_normalization(self.q1,training=self.is_training)
		# self.q2 = tf.layers.batch_normalization(self.q2,training=self.is_training)
		# self.q2 = self.q2_word_rep#+self.q2_char_rep

		self.dense1_w = tf.get_variable(name='dense1_w',shape=[2*self.hidden_dim,self.hidden_dim])
		self.dense1_b = tf.get_variable(name='dense1_b',shape=[self.hidden_dim])
		self.q1 = tf.matmul(self.q1,self.dense1_w) + self.dense1_b
		self.q1 = tf.nn.tanh(self.q1)
		# self.q1 = tf.nn.relu(self.q1)

		self.q2 = tf.matmul(self.q2,self.dense1_w) + self.dense1_b
		self.q2 = tf.nn.tanh(self.q2)

		d = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.q1-self.q2),1)),[-1,1])

		# self.q2 = tf.stop_gradient(self.q2)
		# self.q2 = tf.nn.relu(self.q2)

		# self.q = tf.concat(axis=1,values=[tf.abs(self.q1-self.q2), tf.multiply(self.q1,self.q2)])
		# self.q = tf.concat(axis=1,values=[self.q1,self.q2, tf.multiply(self.q1,self.q2)])
		# self.q = tf.concat(axis=1,values=[self.q1,self.q2, tf.abs(self.q1-self.q2), tf.multiply(self.q1,self.q2),self.q1_score_cnn,self.q2_score_cnn])
		self.q = tf.concat(axis=1,values=[self.q1,self.q2, tf.abs(self.q1-self.q2), tf.multiply(self.q1,self.q2)])
		# self.q = tf.nn.dropout(self.q,self.keep_prob)

		self.dense2_w = tf.get_variable(name='dense2_w',shape=[4*self.hidden_dim,self.hidden_dim])
		self.dense2_b = tf.get_variable(name='dense2_b',shape=[self.hidden_dim])
		self.q = tf.matmul(self.q,self.dense2_w) + self.dense2_b
		# self.q = tf.layers.batch_normalization(self.q,training=self.is_training)
		# self.q = tf.layers.batch_normalization(self.q,training=self.is_training)

		self.q = tf.nn.relu(self.q)
		# self.q = tf.nn.relu(self.q)
		# self.q = tf.nn.dropout(self.q,self.keep_prob)
		

		self.dense3_w = tf.get_variable(name='dense3_w',shape=[self.hidden_dim,1])
		self.dense3_b = tf.get_variable(name='dense3_b',shape=[1])
		out = tf.reshape(tf.add(tf.matmul(self.q,self.dense3_w),self.dense3_b),[-1])   

		# tmp = tf.reshape(tf.constant([-2.0,2.0]),[1,2])
		# values = tf.stop_gradient(tf.reshape(tf.reduce_sum(tf.multiply(tmp,tf.cast(self.label,tf.float32)),1),[-1]))

		# self.loss = tf.cast(tf.reduce_sum(tf.square(values-out)),tf.float64)
		# similary = tf.reshape(tf.div(tf.reshape(tf.reduce_sum(tf.multiply(self.q1,self.q2),1),[-1]), tmp),[-1,1])

		self.merge = tf.reshape(tf.nn.sigmoid(out),[-1,1])
		# # self.merge = tf.clip_by_value(tf.cast(self.merge,tf.float64),1e-15,1.0-1e-15)
		self.non_merge = 1.0-self.merge
		self.prob = tf.concat(axis=1,values=[self.non_merge,self.merge])
		self.prediction = tf.argmax(self.prob,1)

		# weight = tf.reduce_sum(tf.square(self.prob - tf.cast(self.label,tf.float32)),1)

		# weighta = tf.multiply(tf.cast(tf.less(tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1),0.3),tf.float32),1.0)
		# weightb = 1.0 - tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)
		# weight = tf.stop_gradient(weighta+weightb)

		# weight = tf.stop_gradient(tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)+0.3)

		# weight = tf.stop_gradient(1.4 - tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1))
		# weight = tf.square(weight)
		# a = tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)
		# b = tf.cast(tf.less(a,0.8),tf.float32)*0.8 + 0.2
		# weight = tf.multiply(tf.abs(0.9-a),b)
		# weight = 1.0 - tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)
		# weight = tf.stop_gradient(weight)
		# weight=1.0
		# weight = tf.stop_gradient(1.5*tf.cast(tf.argmax(self.label,1),tf.float32)+0.5)
		weight = 1.0
		# b = 0.5+a
		# weight = tf.stop_gradient(b)
		# weight = tf.stop_gradient(1.0/2.0 - tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)/2.0)
		# weight = 1.5 - tf.reduce_sum(tf.multiply(self.prob,tf.cast(self.label,tf.float32)),1)
		# self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(self.prob,tf.float64),1e-15,1.0-1e-15)),tf.multiply(1.0 - tf.stop_gradient(tf.cast(self.prob,tf.float64)), tf.cast(self.label,tf.float64))),1))
		# self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(self.prob,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1))
		# self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(self.prob,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1))
		# self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)))
		# self.loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)),weight))

		margin = 0.0
	
		tmp = tf.stop_gradient(tf.zeros_like(d))

		# prob = 2.0 / (1.0 + tf.exp(d))
		prob = tf.nn.sigmoid(d)
		neg_prob = 1.0 - prob
		loss = tf.concat(axis=1,values=[neg_prob,prob])
		# self.prob = loss
		# self.prediction = tf.argmax(self.prob,1)

		# d2 = tf.reshape(tf.square(d),[-1,1])
		# neg_d = tf.reshape(tf.square(tf.reduce_max(tf.concat(axis=1,values=[tf.reshape(margin-d,[-1,1]), tf.reshape(tmp,[-1,1])]),1)),[-1,1])
		# loss = tf.concat(axis=1,values=[neg_d,d2])
		# self.loss = 0.05*tf.reduce_mean(tf.reduce_sum(tf.multiply(loss,tf.stop_gradient(tf.cast(self.label,tf.float32))),1))
		# self.loss += tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)),weight))

		# self.loss = 0.5*tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)),weight))
		# self.loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)),weight))
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32)))


		# self.loss = tf.cast(self.loss,tf.float64)
		# self.loss += 0.1*tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(loss,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1))
		# self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(loss,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1))

		# self.loss += tf.cast(tf.reduce_mean(tf.square(tf.reshape(self.merge,[-1]) - tf.stop_gradient(tf.reshape(tf.cast(tf.argmax(self.label,1),tf.float32),[-1])))),tf.float64)
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32))
		self.test_accuracy = tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32)
		# self.test_loss  = tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(self.prob,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1)
		self.test_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels = tf.cast(tf.reshape(tf.argmax(self.label,1),[-1]),tf.float32))
		# self.test_loss = tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(loss,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1)


	def deal10(self):
		self.q1 = tf.concat(axis=1,values=[self.q1_word_rep1,self.q1_word_rep2,self.q1_word_cnns])
		self.q2 = tf.concat(axis=1,values=[self.q2_word_rep1,self.q2_word_rep2,self.q2_word_cnns])

		self.q1 = tf.layers.batch_normalization(self.q1,training=self.is_training)
		self.q2 = tf.layers.batch_normalization(self.q2,training=self.is_training)
		# self.q2 = self.q2_word_rep#+self.q2_char_rep

		self.dense1_w = tf.get_variable(name='dense1_w',shape=[5*self.hidden_dim,self.hidden_dim])
		self.dense1_b = tf.get_variable(name='dense1_b',shape=[self.hidden_dim])
		self.q1 = tf.matmul(self.q1,self.dense1_w) + self.dense1_b
		self.q1 = tf.nn.tanh(self.q1)
		# self.q1 = tf.nn.relu(self.q1)

		self.q2 = tf.matmul(self.q2,self.dense1_w) + self.dense1_b
		self.q2 = tf.nn.tanh(self.q2)

		margin = 5.0
	
		d = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.q1-self.q2),1)),[-1])
		tmp = tf.stop_gradient(tf.zeros_like(d))

		d2 = tf.reshape(tf.square(d),[-1,1])
		neg_d = tf.reshape(tf.square(tf.reduce_max(tf.concat(axis=1,values=[tf.reshape(margin-d,[-1,1]), tf.reshape(tmp,[-1,1])]),1)),[-1,1])
		loss = tf.concat(axis=1,values=[neg_d,d2])
		self.loss = tf.reduce_mean(tf.multiply(loss,tf.stop_gradient(tf.cast(self.label,tf.float32))))
		
		# loss1 = tf.multiply(tf.square(d),tf.stop_gradient(tf.cast(tf.argmax(self.label,1),tf.float32)))
		# loss2 = tf.multiply(tf.square(tf.reduce_max(tf.concat(axis=1,values=[tf.reshape(margin-d,[-1,1]), tf.reshape(tmp,[-1,1])]),1)),tf.stop_gradient(tf.cast(tf.argmin(self.label,1),tf.float32)))
		# self.loss = tf.reduce_mean(loss1+loss2)

		# prob = tf.reshape(tf.exp(-d),[-1,1])#
		prob = tf.reshape(tf.reduce_max(tf.concat(axis=1,values=[1.0 - tf.reshape(d/margin,[-1,1]), tf.reshape(tmp,[-1,1])]),1),[-1,1])
		neg_prob = 1.0 - prob
		self.prob = tf.concat(axis=1,values=[neg_prob,prob])

		self.prediction = tf.argmax(self.prob,1)

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32))

		self.test_accuracy = tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32)

		self.test_loss  = tf.reduce_sum(tf.multiply(-tf.log(tf.clip_by_value(tf.cast(self.prob,tf.float64),1e-15,1.0-1e-15)),tf.cast(self.label,tf.float64)),1)

		
	def cnn(self,inputs,embedding,w,b,mask,lens,window,raw=True):

		if raw:
			inputs = tf.nn.embedding_lookup(embedding,inputs)
		input_forward = tf.expand_dims(inputs,-1)
	
		conv = tf.nn.conv2d(
			input_forward,
			w,
			strides=[1,1,1,1],
			padding='VALID', # padding='VALID',
			name='conv'
			)
		h = tf.nn.bias_add(conv,b)

		h = tf.reshape(h,[-1,self.hidden_dim])
		mask = tf.reshape(mask[:,:lens-window+1],[-1,1])
		h = tf.reshape(tf.reduce_max(tf.reshape(h+mask,[-1,lens-window+1,self.hidden_dim]),1),[-1,self.hidden_dim])
	
		# h = tf.nn.tanh(h)
		h = tf.nn.dropout(h,self.keep_prob)
		# pooled = tf.nn.max_pool(
		# 	h,
		# 	ksize=[1,lens-window+1,1,1],
		# 	strides=[1,1,1,1],
		# 	padding='VALID',
		# 	name='pool'
		# 	)
		# h = tf.reshape(pooled,[-1,self.hidden_dim])
		# h = tf.nn.tanh(h)
		# h = tf.nn.dropout(h,self.keep_prob)
		
		return h
	def embed(self,cnn_rep,cell_forward,cell_backward,inputs,att_w,att_b,att_v,lens,mask,mask01,scope='WORD',reuse=False):

		

		# inputs_forward = tf.nn.embedding_lookup(embedding,inputs)
		# inputs_backward = tf.nn.embedding_lookup(embedding,tf.reverse(inputs,[1]))
		inputs_forward = inputs
		with tf.variable_scope(scope+'_LSTM_FORWARD'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward,inputs_forward,dtype=tf.float32)

		output_h = 0.5*tf.add(outputs[0],outputs[1])

		tmp = tf.constant(0.0,shape=[1,lens,2*self.hidden_dim])

		rep = tf.reshape(cnn_rep,[-1,1,2*self.hidden_dim]) - tmp

		rep = -1.0*tf.reduce_sum(tf.multiply(tf.reshape(output_h,[-1,self.hidden_dim]),tf.reshape(rep,[-1,self.hidden_dim])),1)
		rep = tf.reshape(tf.multiply(tf.nn.softmax(tf.reshape(rep,[-1,lens])+mask,1),mask01),[-1,1,lens])
		# rep = tf.concat(axis=2,values=[output_h,rep])
		# rep = tf.reshape(rep,[-1,3*self.hidden_dim])
		# rep = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(rep,att_w) + att_b),att_v),[-1,lens]),1),[-1,1,lens])
		h = tf.reshape(tf.matmul(rep,output_h),[-1,self.hidden_dim])

		# h = tf.layers.batch_normalization(h,training=self.is_training)

		
		# rep = tf.reshape(rep,[-1,2*self.hidden_dim])
		# rep = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(rep,att_w) + att_b),att_v),[-1,lens]),1),[-1,1,lens])
		# h = tf.reshape(tf.matmul(rep,output_h),[-1,self.hidden_dim])
		# h = tf.nn.tanh(h)
		# h = tf.nn.dropout(h,self.keep_prob)
		return h

	

	def embed_l1(self,cell_forward,cell_backward,inputs,embedding,att,lens,mask,mask01,scope='WORD',reuse=False,raw=True):

		

		if raw:
			inputs_forward = tf.nn.embedding_lookup(embedding,inputs)
			
		else:
			inputs_forward = inputs
			

		
	
		with tf.variable_scope(scope+'_LSTM_FORWARD'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward,inputs_forward,dtype=tf.float32)


		output_forward = outputs[0] 
		output_backward = outputs[1]

		# output_h = tf.concat(axis=2,values=[output_forward,output_backward])
		output_h = tf.reshape(tf.multiply(tf.reshape(0.5*tf.add(output_forward,output_backward),[-1,self.hidden_dim]),tf.reshape(mask01,[-1,1])),[-1,lens,self.hidden_dim])
		# h = tf.reshape(tf.reduce_max(output_h,1),[-1,self.hidden_dim])

		h = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h),[-1,self.hidden_dim]),att),[-1,lens])+mask ,1),[-1,1,lens]),output_h),[-1,self.hidden_dim])
		# h = tf.reshape(tf.reduce_max(output_h,1),[-1,self.hidden_dim])
		# h = tf.nn.tanh(h)
		# h = tf.nn.dropout(h,self.keep_prob)
		# h = tf.layers.batch_normalization(h,training=self.is_training)

		return output_h ,h

	def LSTM(self,layers=1):
		lstms = []

		for num in range(layers):

			lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			lstm = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=self.keep_prob)
			lstms.append(lstm)

		lstms = tf.contrib.rnn.MultiRNNCell(lstms)
		return lstms
