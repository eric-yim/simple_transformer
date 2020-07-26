from singleout_transformer import Transformer
from singleout_config import Config
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from singleout_mask_helper import create_masks

#==========================================================================================
# DATA
def load_data():
    c = Config(0)
    return  pd.read_csv(c.data_path,header=None,index_col=False).values    

def load_data_and_initialize():
    global c
    data = load_data()
    pe_input = data.shape[1]-1
    max_num = np.max(data)
    c = Config(max_num,pe_input)
    return data
def create_tf_dataset():
    data = load_data_and_initialize()
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(c.train_kwargs['buffer_size']).padded_batch(c.train_kwargs['batch_size'],padded_shapes=(None,))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset
#==========================================================================================
# MODEL    
def load_model():
    global c,transformer,ckpt_manager
    transformer = Transformer(**c.model_kwargs)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, c.checkpoint_path, max_to_keep=10)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
#==========================================================================================
# MODEL   
def evaluate_batch(inp_sentences):
  encoder_input = inp_sentences
  output = tf.zeros_like(inp_sentences[:,-1:])

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
  # predictions.shape == (batch_size, seq_len, vocab_size)
  predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
  predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
  predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
  return predicted_id, attention_weights
def complete_prompt(inp_sequence,max_len=8):
    inp_sequence = np.array(inp_sequence,dtype=np.int32)[np.newaxis,...]
    while inp_sequence.shape[-1] < max_len:
        
        next_id ,_ = evaluate_batch(inp_sequence)
        inp_sequence = np.concatenate([inp_sequence,next_id.numpy()],axis=-1)

    return inp_sequence

def main():
    data = load_data_and_initialize()
    load_model()
    
    #BATCHES OF TRAINING
    inp = data[...,:-1]
    aa,_=evaluate_batch(inp)
    print("\nTRAINING SAMPLES")
    for a,d in zip(aa.numpy(),inp):
        print("\nOrg:",d)
        print("Res:",a)
        
    #SINGLE METHOD
    print("\nPrompt Completions")
    data = [3,4]
    completed = complete_prompt(data)
    print("Input Prompt: {}".format(data))
    print("Completed Prompt: {}".format(completed))
    
    
main()
#print(a)
