import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
from scipy import sparse
checkpoint_path = os.path.join("/home/huihui/tflight/tsv2/model-pb", "pcc_model_1.ckpt")
meta_path = os.path.join("/home/huihui/tflight/tsv2/model-pb", "pcc_model_1.ckpt.meta")
#reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
# 	print("tensor_name: ", key)
# 	print(reader.get_tensor(key))
# 	# x=reader.get_tensor(key)
# 	# print("X: ",x)

#import graph
saver = tf.train.import_meta_graph(meta_path)

pruning_perc=25
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	#Import variable values
	saver.restore(sess,checkpoint_path)
	graph = tf.get_default_graph()
	trainable_variables = tf.trainable_variables()
	pruned_weights = [[]]
	for v in tf.trainable_variables():
		print(v.name,"v.shape:",v.shape)
		if "w:0" in v.name:
			pruned_weights=v
			print("pruned_weights",pruned_weights)
			mask = np.ones(v.shape)
			X = v.shape[0]
			Y = v.shape[1]
			# computing L2-norm of weight matrix
			unit_norm = np.linalg.norm(sess.run(v), axis=0)
			# calculate threshold based on sparsity percentage
			threshold = np.percentile(unit_norm, pruning_perc)
			ids = unit_norm < threshold
			count = 0
			for i in ids:
				if i== True:
					count+=1
			print("count",count)
			# setting the columns to zero
			mask[:, ids] = 0
			# setting the mask onto the weight matrix
			sess.run(v.assign(tf.multiply(v, mask)))
	saver.save(sess,"/home/huihui/tflight/tsv2/model-new/pcc_model.ckpt")

checkpoint_path_new = os.path.join("/home/huihui/tflight/tsv2/model-new", "pcc_model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path_new)
var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
# 	print("tensor_name: ", key,"shape:",key.shape)
# 	print(reader.get_tensor(key))
	# x=reader.get_tensor(key)
	# print("X: ",x)

