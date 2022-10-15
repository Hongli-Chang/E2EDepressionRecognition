'''
2020.04.22
@lsy
Function:
    To evaluate our model on SEED.
    Model: combine IAG and SGA.

    Subject Independent.
'''

from __future__ import absolute_import, division, print_function

import argparse
import ast
import os
import time

import numpy as np
import scipy.io as scio
import tensorflow as tf
from dataProcess import DataGenerate
from model import graph_lstm
from sklearn.metrics import confusion_matrix
from tensorflow import set_random_seed

def readData(data_txt, SUB):
    data, label, subject = [], [], []
    with open(data_txt) as f:
        sub = 0
        lines = f.readlines()
        for l in lines:
            sub += 1
            if sub not in SUB:
                continue
            dataIn = scio.loadmat(l.split()[0])
            tempData = np.asarray(dataIn['data'], dtype=np.float32)
            tempLabel = np.asarray(dataIn['label'], dtype=np.float32)
            tempSub = sub * np.ones(tempLabel.shape[0])
            if not len(data):
                data, label, subject = tempData, tempLabel, tempSub
            else:
                data = np.concatenate((data, tempData), axis=0)
                label = np.concatenate((label, tempLabel), axis=0)
                subject = np.concatenate((subject, tempSub), axis=0)
        label = label.reshape(label.shape[0])
    return data, label, subject


def train(args, txtName, p_dir):
    if args.isSeed:
        np.random.seed(18)    
        set_random_seed(23)

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    if not os.path.exists(args.logDir):
        os.makedirs(args.logDir)

    seq_length = 62
    x = tf.placeholder(tf.float32, [None, 62, 5])
    y = tf.placeholder(tf.int64, [None])
    train_flag = tf.placeholder(tf.bool)
    y_pre, alpha = graph_lstm(x, seq_length, train_flag, args)
    global_steps = tf.Variable(0, trainable = False)

    with tf.name_scope('loss'):
        tv = tf.trainable_variables()
        alpha_loss = tf.reduce_sum(alpha) * args.lambda_for_att_alpha
        regu_loss = args.lambdaa * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name])
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pre)
        loss = tf.reduce_mean(cross_entropy) + regu_loss + alpha_loss

    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer(args.lr).minimize(loss, global_step=global_steps)

    with tf.name_scope('accuracy'):
        pre_label = tf.argmax(y_pre, axis=1)
        correct_prediction = tf.equal(pre_label, y)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('save'):
        saver = tf.train.Saver()
    
    # ===============================================================
    start = time.time()
    SUB = [1, 5, 8, 10, 13, 16, 19, 23, 25, 30, 33, 36, 39, 41, 45]
    data, label, subject = readData(args.dataDir, SUB)
    
    all_sub_acc = []
    all_c_matrix = [] 
    for n in range(15):
        sub_p_dir = os.path.join(p_dir, 'sub%d'%SUB[n])
        dataInfo = DataGenerate(data = data,
                              label = label, 
                              subject = subject,
                              testSub = SUB[n],
                              batch_size = args.batch_size)
        
        BATCH = dataInfo.batch
        test_data, test_label = dataInfo.test_data, dataInfo.test_label
        acc_max, acc_max_step = 0, 0
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(args.epoch):
                loss_epoch, test_acc = 0, 0

                for j in range(BATCH):
                    train_data, train_label = dataInfo.next_batch()
                    _, loss_batch = sess.run([train_op, loss], 
                                                feed_dict = {x:train_data, y:train_label, train_flag:True})
                    loss_epoch += loss_batch
                    if sess.run(global_steps) % 5 == 0:
                        test_pre, test_acc = sess.run([pre_label, accuracy], 
                                                        feed_dict = {x:test_data, y:test_label, train_flag:True})
                        if test_acc > acc_max:
                            (acc_max, acc_max_step) = (test_acc, sess.run(global_steps))
                            c_matrix = confusion_matrix(test_label, test_pre)
                            _alpha = sess.run([alpha],
                                                feed_dict={x:test_data, y:test_label, train_flag:False})
                            scio.savemat(sub_p_dir, {'alpha':_alpha})

                print ('subject = %02d, epoch = %03d, test_acc = %-10.4f,  best_acc = %-10.3f, step = %06d, epoch_loss = %f,' 
                        %(SUB[n], i, test_acc, acc_max, acc_max_step, loss_epoch))

                if i % 5 == 0:
                    with open(txtName, 'a+') as t_f:
                        t_f.write('\nsub = ' + str(SUB[n]) + ', epoch = ' + str(i) + ', test_acc = ' + str(test_acc) + ', loss = '   
                                + str(loss_epoch)[0:10] + ', best_acc = ' + str(acc_max) + ', best_step = ' + str(acc_max_step))            
                if acc_max == 1:
                    break
            test_acc = accuracy.eval(feed_dict={x:test_data, y:test_label, train_flag:False})
            acc_max = max(acc_max, test_acc)
            print ('subject = %02d, epoch = %03d, test_acc = %-10.4f,  best_acc = %-10.3f, step = %06d, epoch_loss = %f,' 
                    %(SUB[n], i, test_acc, acc_max, acc_max_step, loss_epoch))
            
            
        all_sub_acc.append(acc_max)
        all_c_matrix.append(c_matrix)
        with open(txtName, 'a+') as t_f:
            t_f.write('\nsub = ' + str(SUB[n]) + ', epoch = ' + str(i) + ', test_acc = ' + str(test_acc) + ', loss = '   
                        + str(loss_epoch)[0:10] + ', best_acc = ' + str(acc_max) + ', best_step = ' + str(acc_max_step))
            t_f.write('\nconfusion matrix is:\n' + str(c_matrix))
            t_f.write('\n***********************************************************')

    end = time.time()
    acc_mean = round(sum(all_sub_acc) / 15, 4) * 100
    acc_std = round(np.std(all_sub_acc), 4) * 100
    all_matrix = sum(all_c_matrix)

    print('***********************************************************')
    print('time is {}'.format(time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
    print('mean/std = {}/{}, time is {}'.format(acc_mean, acc_std ,time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
    
    with open(txtName, 'a+') as t_f:
        t_f.write('\n\ntime is: ' + time.strftime('%Y%m%d_%H:%M:%S', time.localtime()))
        t_f.write('\n\nconfusion matrix is:\n' + str(all_matrix))
        t_f.write('\nmean/std acc = %.2f/%.2f'%(acc_mean, acc_std))
            
