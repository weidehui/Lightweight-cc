import gym
import network_sim0
import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
from scipy import sparse
from afprunenetwork import StuMlpPolicy
from stable_baselines import PPO1
from tflight.tsv2.common.simple_arg_parse import arg_or_default

checkpoint_path = os.path.join("/home/huihui/Lightweight-cc/tflight/tsv2/big-model/data", "pcc_model_8.ckpt")
meta_path = os.path.join("/home/huihui/Lightweight-cc/tflight/tsv2/big-model/data", "pcc_model_8.ckpt.meta")

saver = tf.train.import_meta_graph(meta_path)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

pruning_perc = 25
variables_big = []
def prune(v,ids,sess):
    mask = np.ones(v.shape)
    X = v.shape[0]
    Y = v.shape[1]
    # computing L2-norm of weight matrix
    unit_norm = np.linalg.norm(sess.run(v), axis=0)
    # calculate threshold based on sparsity percentage
    threshold = np.percentile(unit_norm, pruning_perc)
    ids = unit_norm < threshold
    count = 0
    # setting the columns to zero
    mask[:, ids] = 0
    # setting the mask onto the weight matrix
    sess.run(v.assign(tf.multiply(v, mask)))
    v = sess.run(v)
    return v,ids

with tf.Session(config=config) as sess:
    # Import variable values
    saver.restore(sess, checkpoint_path)
    graph = tf.get_default_graph()
    trainable_variables = tf.trainable_variables()
    ids_vf=[]
    ids_pi=[]
    ids=[]
    ids_q=[]
    for v in tf.trainable_variables():
        #print(v.name, "v.shape:", v.shape)
        #print(sess.run(v))
        if "w:0" in v.name:
            name = v.name
            if "pi" in name and "vf" not in name and "q" not in name:
                if "fc0" in name:
                    v, ids_pi= prune(v, ids, sess)
                    v = np.delete(v, ids_pi, axis=1)
                    ids=ids_pi
                else:
                    ids=ids_pi
                    v, ids_pi = prune(v, ids, sess)
                    v = np.delete(v, ids_pi, axis=1)
                    v = np.delete(v, ids, axis=0)
                    ids = ids_pi
            elif "vf" in name:
                if "fc0" in name:
                    v, ids_vf= prune(v, ids, sess)
                    v = np.delete(v, ids_vf, axis=1)
                    ids = ids_vf
                else:
                    ids=ids_vf
                    v, ids_vf = prune(v, ids, sess)
                    v = np.delete(v, ids_vf, axis=1)
                    v = np.delete(v, ids, axis=0)
                    ids = ids_vf
                    if(len(ids)==16):
                        ids_q = ids
            elif "q" in name:
                v, ids = prune(v, ids, sess)
                v = np.delete(v, ids_q, axis=0)
            variables_big.append(v)
        elif "b:0" in v.name: #bias layer
            mask = np.ones(v.shape)
            mask[ids] = 0
            sess.run(v.assign(tf.multiply(v, mask)))
            v = sess.run(v)
            v = np.delete(v, ids)
            variables_big.append(v)
            #print(v)
        else:
            v = sess.run(v)
            variables_big.append(v)
        print("name:",name," shape:",v.shape)

training_sess = None
env = gym.make('PccNs-v0')
gamma = arg_or_default("--gamma", default=0.99)
model = PPO1(StuMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)


with model.graph.as_default():
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        #saver.save(training_sess, "./pcc_model_%d.ckpt")
        trainable_variables = tf.trainable_variables()
        i = 0
        temp=variables_big
        for v in trainable_variables:
            #print("name:",v.name," shape:",v.shape,"temp[%d]:"%i,temp[i])
            print("name:", v.name, " shape:", v.shape)
            sess.run(v.assign(temp[i]))
            i+=1

        saver.save(sess, "/home/huihui/Lightweight-cc/tflight/tsv2/model-new/pcc_model_new.ckpt")






















