# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from clean_str import clean_str
from nlgeval import NLGEval
from calculate_ter import calculate_ter
from seq2seq_syn import seq2seq

def cal_scores():
    return 0

def finetune_model(num_epochs=20, k_fold=10,predict_test=True,calculate_scores=False):
    model = seq2seq(mode='finetune')
    saver = tf.train.Saver()
    tf.logging.set_verbosity(logging.INFO)
    config = tf.ConfigProto()
    
    config.gpu_options.allow_growth = True
    
    #Reading data from csv file
    data = pd.read_csv(model.active_passive_csv_file,usecols=[2,3])
    print(data.head())
    data_clean = data.applymap(clean_str)
    
    if calculate_scores is True:
        nlgeval = NLGEval(no_skipthoughts=True,no_glove=False)
        #bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [],[],[],[]
    bleu_1_scores_top_beam, bleu_2_scores_top_beam, bleu_3_scores_top_beam, bleu_4_scores_top_beam = [],[],[],[]
    meteor_scores_top_beam = []
    bleu_4_best_of_10_result_scores, meteor_best_of_10_result_scores = [],[]
    rouge_scores_top_beam, rouge_scores_best_of_10_result_scores = [],[]
    greedy_embedding_top_beam, greedy_embedding_best_of_10_result_scores = [],[]
    ter_top_beam, ter_best_of_10_result_scores_with_best_bleu, ter_best_of_10_result_scores_with_best_meteor = [],[],[]
    
    for i in range(k_fold):
        iter_no=0
        train,test = train_test_split(data_clean,test_size=0.2)
        print('\nTrain head: ',train.head())
        print('\n\nTest head: ',test.head())
        next_batch = model.train_batchify_pandas(train)
        
        print('\n\nRetraining again for #{} validations----------------------------------\n\n'.format(i))
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(model.model_dir,model.model_dir)))
            if ckpt and ckpt.model_checkpoint_path:
                print("Found checkpoints. Restoring variables..")
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("Checkpoints restored...")
                
            for epoch in range(num_epochs):
                train_cost = 0
                train_op = model.get_train_op(epoch)
                print('Training op --- ', train_op)
                for _ in range(model.samples//model.batch_size):
                    iter_no = iter_no + 1
                    batch = next_batch()
                    feed = batch
                    feed[model.iter_no]=iter_no
                    v = sess.run([train_op,model.loss,model.global_step],batch)
                    train_cost += v[1]
                    print('Global Step: ',v[2],', Step Loss: ',v[1])
    #                print(v[3][-1][-1])
                
                train_cost /= model.samples / model.batch_size
                print("Epoch",(epoch+1),train_cost)
                
            print("Saving model at "+str(v[2])+" iterations...")
            saver.save(sess,os.path.join(model.new_dir,'model.ckpt'),global_step=model.global_step)
            print("Model saved..")
            
            #Running for test split and compute BLEU scores
            test_next_batch = model.train_batchify_pandas(test)
            hyp_top_beam_result, hyp_best_of_10_beam_result, ref = [],[],[]
            hyp_best_of_10_meteor_result, hyp_best_of_10_rouge_result = [],[]
            hyp_best_of_10_greedy_emb_result = []
            
    #        hyp, ref = [],[]
            print("Predictions of test batch: \n")
            for _ in range(len(test)//model.batch_size):
                test_batch = test_next_batch()
                temp_ref = [model.to_str(string).replace('</S>','').replace('<S>','').strip() for string in test_batch[model.outputs]]
                ref.extend(temp_ref)
                
                v = sess.run([model.global_step,model.predictions],test_batch)
                for i,pred in enumerate(v[1]['preds']):
                    max_bleu = 0.0
                    max_meteor = 0.0
                    max_rouge = 0.0
                    max_greedy_emb = 0.0
                    
                    best_bleu_str = ''
                    best_meteor_str = ''
                    best_rouge_str = ''
                    best_greedy_emb_str = ''
                    
                    for j in range(model.beam_width):
                        predicted = model.to_str(pred[:,j]).replace('</S>','').replace('<S>','').strip()
                        if j==0:
                            hyp_top_beam_result.append(predicted)
                        
                        metrics_dict = nlgeval.compute_individual_metrics([temp_ref[i]],predicted)
                        if metrics_dict['Bleu_4'] >= max_bleu:
                            max_bleu = metrics_dict['Bleu_4']
                            best_bleu_str = predicted
                        
                        if metrics_dict['METEOR'] >= max_meteor:
                            max_meteor = metrics_dict['METEOR']
                            best_meteor_str = predicted
                        
                        if metrics_dict['ROUGE_L'] >= max_rouge:
                            max_rouge = metrics_dict['ROUGE_L']
                            best_rouge_str = predicted
                        
                        if metrics_dict['GreedyMatchingScore'] >= max_greedy_emb:
                            max_greedy_emb = metrics_dict['GreedyMatchingScore']
                            best_greedy_emb_str = predicted
                            
                    hyp_best_of_10_beam_result.append(best_bleu_str)
                    hyp_best_of_10_meteor_result.append(best_meteor_str)
                    hyp_best_of_10_rouge_result.append(best_rouge_str)
                    hyp_best_of_10_greedy_emb_result.append(best_greedy_emb_str)
                    
                    """
                    predicted = model.to_str(seq).replace('</S>','').replace('<S>','').strip()
    #                print(i,predicted)
                    hyp.append(predicted)
                    """
            
    #        print(ref,hyp)
            print('Ref len: ',len(ref),', Hyp len: ',len(hyp_top_beam_result))
            metrics_dict_top_beam_result = nlgeval.compute_metrics([ref],hyp_top_beam_result)
            metrics_dict_best_of_5_beam = nlgeval.compute_metrics([ref],hyp_best_of_10_beam_result)
            metrics_dict_best_of_5_meteor = nlgeval.compute_metrics([ref],hyp_best_of_10_meteor_result)
            metrics_dict_best_of_5_rouge = nlgeval.compute_metrics([ref],hyp_best_of_10_rouge_result)
            metrics_dict_best_of_5_greedy_emb = nlgeval.compute_metrics([ref],hyp_best_of_10_greedy_emb_result)
            
            ter_top = calculate_ter(ref,hyp_top_beam_result)
            ter_bleu = calculate_ter(ref,hyp_best_of_10_beam_result)
            ter_meteor = calculate_ter(ref,hyp_best_of_10_meteor_result)
            
            print("BLEU 1-4 : ",metrics_dict_top_beam_result['Bleu_1'],metrics_dict_top_beam_result['Bleu_2'],
                  metrics_dict_top_beam_result['Bleu_3'],metrics_dict_top_beam_result['Bleu_4'])
            
            print("METEOR, ROUGE_L, GreedyEmbedding : ",metrics_dict_top_beam_result['METEOR'],
                  metrics_dict_top_beam_result['ROUGE_L'],metrics_dict_top_beam_result['GreedyMatchingScore'])
            
            print("CIDER, EmbeddingAverageCosineSimilaity, VectorExtremaCosineSimilarity : ",
                  metrics_dict_top_beam_result['CIDEr'],
                  metrics_dict_top_beam_result['EmbeddingAverageCosineSimilairty'],
                  metrics_dict_top_beam_result['VectorExtremaCosineSimilarity'])
            print("TER top beam, TER Bleu, TER METEOR",ter_top,ter_bleu,ter_meteor)
            
    #        metrics_dict = nlgeval.compute_metrics([ref],hyp)
            # Top beam search results
            bleu_1_scores_top_beam.append(metrics_dict_top_beam_result['Bleu_1'])
            bleu_2_scores_top_beam.append(metrics_dict_top_beam_result['Bleu_2'])
            bleu_3_scores_top_beam.append(metrics_dict_top_beam_result['Bleu_3'])
            bleu_4_scores_top_beam.append(metrics_dict_top_beam_result['Bleu_4'])
            meteor_scores_top_beam.append(metrics_dict_top_beam_result['METEOR'])
            rouge_scores_top_beam.append(metrics_dict_top_beam_result['ROUGE_L'])
            greedy_embedding_top_beam.append(metrics_dict_top_beam_result['GreedyMatchingScore'])
            ter_top_beam.append(ter_top)
            
            # Best of 5 search results
            ter_best_of_10_result_scores_with_best_bleu.append(ter_bleu)
            ter_best_of_10_result_scores_with_best_meteor.append(ter_meteor)
            bleu_4_best_of_10_result_scores.append(metrics_dict_best_of_5_beam['Bleu_4'])
            meteor_best_of_10_result_scores.append(metrics_dict_best_of_5_meteor['METEOR'])
            rouge_scores_best_of_10_result_scores.append(metrics_dict_best_of_5_rouge['ROUGE_L'])
            greedy_embedding_best_of_10_result_scores.append(metrics_dict_best_of_5_greedy_emb['GreedyMatchingScore'])
            
            
    print('Bleu 1 scores top: ',bleu_1_scores_top_beam)
    print('Bleu 2 scores top: ',bleu_2_scores_top_beam)
    print('Bleu 3 scores top: ',bleu_3_scores_top_beam)
    print('Bleu 4 scores top: ',bleu_4_scores_top_beam)
    print('METEOR scores top: ',meteor_scores_top_beam)
    print('ROUGE scores top: ',rouge_scores_top_beam)
    print('Greedy Embedding top: ',greedy_embedding_top_beam)
    print('Translation Edit Rate top: ',ter_top_beam)
    print('\n\nBest beam results:---------\n')
    print('Bleu 4 scores best beam : ',bleu_4_best_of_10_result_scores)
    print('METEOR scores best beam : ',meteor_best_of_10_result_scores)
    print('ROUGE scores best beam: ',rouge_scores_best_of_10_result_scores)
    print('Greedy Embedding best beam: ',greedy_embedding_best_of_10_result_scores)
    print('Translation Edit Rate Bleu: ',ter_best_of_10_result_scores_with_best_bleu)
    print('Translation Edit Rate METEOR: ',ter_best_of_10_result_scores_with_best_meteor)

if __name__=='__main__':
    finetune_model()