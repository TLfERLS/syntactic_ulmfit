# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
import numpy as np
from seq2seq_syn import seq2seq
from clean_str import clean_str

def check_no_lines(filename):
    with open(filename,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        no_of_lines = len(lines)
    return no_of_lines

def load_and_predict(model_dir=None):
    model = seq2seq(mode='predict')
    saver = tf.train.Saver()
    #Reading Active Passive sentences from CSV
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    no_of_lines = check_no_lines(model.test_filename)
    
    
    if no_of_lines < model.batch_size:
        model.batch_size = no_of_lines
        
    next_batch = model.test_batchify()
    
    tf.logging.set_verbosity(logging.INFO)
    
    # Commented in the __init__ function and moved here, for creating vocabulary for a different training set
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(model.model_dir,model.model_dir)))
        
        if ckpt and ckpt.model_checkpoint_path:
            print("Found checkpoints. Restoring variables from checkpoint:\n",ckpt)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("Checkpoints restored...")
        else:
            print('Found no checkponts')
        
        outs = 0
        print("Predictions of test batch: \n")
        for k in range((int)(np.ceil(no_of_lines/model.batch_size))):
            if k*model.batch_size < no_of_lines and (k+1)*model.batch_size > no_of_lines:
                test_batch = next_batch(no_of_lines%model.batch_size)
            else:
                test_batch = next_batch(model.batch_size)
            
            v = sess.run([model.global_step,model.predictions],test_batch)
            for i,pred in enumerate(v[1]['preds']):
                outs = outs+1
                print("\n\nPredictions for sentence #{}:".format(outs))
                # Just checking with top 5 beam search results
                for j in range(model.beam_width):
                    predicted = model.to_str(pred[:,j]).strip()
                    if predicted == '':
                        predicted = '<UNK>'
                    predicted = predicted.replace('<S>','').replace('</S>','')
                    print(predicted,v[1]['scores'][i][j])

if __name__=='__main__':
    model_dir = 'quora_finetuned_enc_dec_active'
    load_and_predict()