# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from clean_str import clean_str
import math

class seq2seq:
    def __init__(self,mode='train',model_dir=None,finetune_dir=None):
        self.num_units = 512
        self.embed_dim = 512
        self.learning_rate = 1.0
        self.optimizer = 'SGD'
        self.batch_size = 8
        self.print_every = 100
        self.iterations = 1250
        self.model_dir = 'quora_clean_model_100k'
        self.finetune_dir = 'quora_finetuned_enc_dec_active'
        self.input_max_length = 30
        self.output_max_length = 30
        self.use_residual_lstm = True
        self.samples = 3200
        self.beam_width = 10
        self.active_passive_csv_file = './data/Active-Passive/Active_Passive_Sentences.csv'
        self.test_filename = './data/Active-Passive/test_source_clean_latest_1.txt'
        self.train_input_filename = './data/Active-Passive/test_source_clean_latest.txt'
        self.train_output_filename = './data/Active-Passive/test_target_clean_latest.txt'
        self.vocab_filename = './data/Active-Passive/train_vocab_clean.txt'
        
        if model_dir is not None:
            self.model_dir = model_dir
        
        if finetune_dir is not None:
            self.finetune_dir = finetune_dir
            
        if mode=='train' or mode=='finetune':
            self.dropout = 0.5
            self.mode = 'train'
        else:
            self.dropout = 0.0
            self.mode = 'predict'
            
        self.create_vocab()
        self.make_inputs()
        self.build_graph()
        
        if mode=='finetune':
            self.create_finetune_ops()
            
    def create_vocab(self):
        self.vocab = {}
        self.rev_vocab = {}
        self.END_TOKEN = 1 
        self.UNK_TOKEN = 2
        with open(self.vocab_filename,encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.vocab[line.strip()] = idx
                self.rev_vocab[idx] = line.strip()
        self.vocab_size = len(self.vocab)
    
    def make_inputs(self):
        self.inputs = tf.placeholder(tf.int64,(None,None))
        self.outputs = tf.placeholder(tf.int64,(None,None))
        self.keep_prob = tf.placeholder(tf.float32)
    
    def tokenize_and_map(self,line):
        s = [self.vocab.get(token.strip(), self.UNK_TOKEN) for token in line.split(' ')]
        return s
    
    def to_str(self,sequence):
            tokens = [
                self.rev_vocab.get(x, "<UNK>") for x in sequence]
            return ' '.join(tokens)
    
    def train_batchify_pandas(self,train):
        def sampler(train):
            i=0
            mark = len(train)
            while True:
                if i==mark:
                    i=0
                else:
                    i=i+1
                    yield {'input':self.tokenize_and_map(train.iloc[i-1][0])[:self.input_max_length - 1] + [self.END_TOKEN],
                           'output':self.tokenize_and_map(train.iloc[i-1][1])[:self.output_max_length - 1] + [self.END_TOKEN]}
        
        data_sampler = sampler(train)
        def next_batch():
            source, target = [],[]
            input_length, output_length = 0,0
            
            for i in range(self.batch_size):
                rec = data_sampler.__next__()
                source.append(rec['input'])
                target.append(rec['output'])
                input_length = max(input_length,len(source[-1]))
                output_length = max(output_length,len(target[-1]))
            for i in range(self.batch_size):
                source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                target[i] += [self.END_TOKEN] * (output_length - len(target[i]))
            
            batch = {
                    self.inputs: source,
                    self.outputs: target,
                    self.keep_prob: 1. - self.dropout
                    }
            
            return batch
        return next_batch
    
    def test_batchify(self):
        def sampler():
            while True:
                with open(self.test_filename,'r',encoding='utf-8') as finput:
                    for source in finput:
                        yield {
                                'input': self.tokenize_and_map(clean_str(source))[:self.input_max_length - 1] + [self.END_TOKEN]}
        
        data_feed = sampler()
        def next_batch(batch_size=8):
            source, target = [],[]
            input_length, output_length = 0,0
            
            for i in range(batch_size):
                rec = data_feed.__next__()
                source.append(rec['input'])
                
                if self.mode == 'train':
                    target.append(rec['output'])
                    output_length = max(output_length,len(target[-1]))
                    
                input_length = max(input_length,len(source[-1]))
                
            for i in range(batch_size):
                source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                if self.mode == 'train':
                    target[i] += [self.END_TOKEN] * (output_length - len(target[i]))
            
            batch = {
                    self.inputs: source,
                    self.outputs: source,
                    self.keep_prob: 1. - self.dropout
                    }
            
            return batch
        return next_batch
    
    def get_dropout_cell(self,lstm_size,keep_prob):
        lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    def get_basic_cell(self,lstm_size):
        lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
        return lstm
    
    def build_graph(self):
        
        batch_size = tf.shape(self.inputs)[0]
        start_tokens = tf.zeros([batch_size], dtype=tf.int64)
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), self.outputs], 1)
        input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.inputs, 1)), 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
        input_embed = layers.embed_sequence(self.inputs, vocab_size=self.vocab_size, embed_dim = self.embed_dim, scope='embed')
        output_embed = layers.embed_sequence(train_output, vocab_size=self.vocab_size, embed_dim = self.embed_dim, scope='embed', reuse=True)
        
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')
        
        residual_cell_1 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,self.keep_prob) for i in range(2)])
        if self.use_residual_lstm:
            residual_cell_1 = tf.contrib.rnn.ResidualWrapper(residual_cell_1)
        residual_cell_2 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,self.keep_prob) for i in range(2)])
        if self.use_residual_lstm:
            residual_cell_2 = tf.contrib.rnn.ResidualWrapper(residual_cell_2)
        final_encoder_cell = tf.contrib.rnn.MultiRNNCell([residual_cell_1, residual_cell_2])
        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(final_encoder_cell, input_embed, dtype=tf.float32)
        print("Encoder created !!!")
        
        def decode(helper,scope,reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                residual_decoder_cell_1 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,self.keep_prob) for i in range(2)])
                if self.use_residual_lstm:
                    residual_decoder_cell_1 = tf.contrib.rnn.ResidualWrapper(residual_decoder_cell_1)
                residual_deocder_cell_2 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,self.keep_prob) for i in range(2)])
                if self.use_residual_lstm:
                    residual_deocder_cell_2 = tf.contrib.rnn.ResidualWrapper(residual_deocder_cell_2)
                final_decoder_cell = tf.contrib.rnn.MultiRNNCell([residual_decoder_cell_1,residual_deocder_cell_2])
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(final_decoder_cell, self.vocab_size, reuse=reuse)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,initial_state=self.encoder_final_state)
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=self.output_max_length)
                return outputs[0]
        
        def predict_decode(scope,reuse=True):
            with tf.variable_scope(scope,reuse=reuse):
#                tiled_seq_length = tf.contrib.seq2seq.tile_batch(input_lengths,multiplier=beam_width)
#                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_outputs,multiplier=beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_final_state,multiplier=self.beam_width)
                
                residual_decoder_cell_1 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,1.0-self.dropout) for i in range(2)])
                if self.use_residual_lstm:
                    residual_decoder_cell_1 = tf.contrib.rnn.ResidualWrapper(residual_decoder_cell_1)
                residual_deocder_cell_2 = tf.contrib.rnn.MultiRNNCell([self.get_dropout_cell(self.num_units,1.0-self.dropout) for i in range(2)])
                if self.use_residual_lstm:
                    residual_deocder_cell_2 = tf.contrib.rnn.ResidualWrapper(residual_deocder_cell_2)
                final_decoder_cell = tf.contrib.rnn.MultiRNNCell([residual_decoder_cell_1,residual_deocder_cell_2])
#                attn_cell = tf.contrib.seq2seq.AttentionWrapper(final_decoder_cell, attention_mechanism, attention_layer_size=num_units/2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(final_decoder_cell, self.vocab_size, reuse=reuse)
                decoder_initial_state = tiled_encoder_final_state
                predict_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=out_cell,embedding=embeddings,
                        start_tokens=tf.tile(tf.constant([0],dtype=tf.int32),[batch_size]),
                        end_token=1,
                        initial_state=decoder_initial_state,
                        beam_width=self.beam_width)
                pred_decode_output,final_state,_ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = predict_decoder,
                        impute_finished=False,
                        maximum_iterations=self.output_max_length)
                return pred_decode_output,final_state
            
        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
        train_outputs = decode(train_helper, 'decode')
#        pred_outputs = decode(pred_helper, 'decode', reuse=True)
        pred_outputs,pred_states = predict_decode('decode',reuse=True)
        
        weights = tf.to_float(tf.not_equal(train_output[:,:-1],1))
        self.global_step = tf.train.get_or_create_global_step()
        self.loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, self.outputs, weights=weights)
        variables = [var.name for var in tf.trainable_variables()]
        print("Variables: ",variables,', length: ',len(variables))
        self.predictions = {'preds':pred_outputs[0],'scores':pred_states[1]}
    
    def create_finetune_ops(self):
#        variables = [var.name for var in tf.trainable_variables()]
        self.variables = tf.trainable_variables()
#        print("Variables: ",variables)
        
        mul = 2.6
        self.iter_no = tf.placeholder(tf.float32, shape=[],name="iter_no")
        
        # Discriminative fine-tuning and slanted triangular learning rate
        self.learing_rate_list = [self.get_learning_rate(0.1/math.pow(mul,i)) for i in range(5)]
        self.optimizers_list = [tf.train.GradientDescentOptimizer(self.learing_rate_list[i]) for i in range(5)]
        grads_deocder = tf.gradients(self.loss,self.variables[9:])
        grads_encoder = tf.gradients(self.loss,self.variables[:9])
        
        # Clubbing(list of list) 2 gradients together corresponding to kernel and bias of a single layer
        self.grads_list_decoder = [grads_deocder[i-2:i] for i in range(2,11,2)]
        
        # For the encoder clubbing 2 gradients corresponding to kernel and bias of every layer except embedding
        self.grads_list_encoder = [grads_encoder[:i+1] if i==0 else grads_encoder[i-1:i+1] for i in range(0,9,2)]
        
        #Keeping the last layer first in list for the learning rate
        self.grads_list_decoder.reverse()
        self.grads_list_encoder.reverse()
        
        self.train_op_list_decoder = []
        self.train_op_list_encoder = []
        
        # Adding the training operations of each lstm layer seperately for decoder.
        # Helpful for gradual unfreezing
        for i,j in enumerate(range(0,-9,-2)):
            if j == 0:
                self.train_op_list_decoder.append(self.optimizers_list[i].apply_gradients(zip(self.grads_list_decoder[i],self.variables[j-2:]),global_step=self.global_step))
            else:
                self.train_op_list_decoder.append(self.optimizers_list[i].apply_gradients(zip(self.grads_list_decoder[i],self.variables[j-2:j])))
        
        for i,j in enumerate(range(7,0,-2)):
            self.train_op_list_encoder.append(self.optimizers_list[i].apply_gradients(zip(self.grads_list_encoder[i],self.variables[j:j+2])))
        # Training op for the embedding
        self.train_op_list_encoder.append(self.optimizers_list[4].apply_gradients(zip(self.grads_list_encoder[4],self.variables[:1])))
    