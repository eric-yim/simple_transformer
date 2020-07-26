from singleout_transformer import Transformer
from singleout_config import Config
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from singleout_mask_helper import create_masks
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
def stepize_data(sequences):
    input_sequences = []
    labels = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i+1]
            input_sequences.append(n_gram_sequence[:-1])
            labels.append(n_gram_sequence[-1])
    max_sequence_len = max([len(x) for x in input_sequences])
    padded = pad_sequences(input_sequences,maxlen=max_sequence_len,truncating='pre',padding='post')
    labels = np.array(labels)[...,np.newaxis]
    return padded,labels
def create_tf_dataset():
    data = load_data_and_initialize()
    data = stepize_data(data)
    train_dataset = tf.data.Dataset.from_tensor_slices(data).map(lambda x,y:(tf.cast(x,tf.int64),tf.cast(y,tf.int64)) )
    
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(c.train_kwargs['buffer_size']).batch(c.train_kwargs['batch_size'])#.padded_batch(c.train_kwargs['batch_size'],padded_shapes=(None,))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset
#==========================================================================================
# MODEL
def load_model():
    global c,transformer,ckpt_manager
    transformer = Transformer(**c.model_kwargs)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, c.checkpoint_path, max_to_keep=10)
#==========================================================================================
# OPTIMIZER
def get_optimizer():
    global c,optimizer
    if c.custom_lr:
      learning_rate = get_lr()
      optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=10):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
def get_lr():
    global c
    return CustomSchedule(c.model_kwargs['d_model'])

#==========================================================================================
# LOSS
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
#==========================================================================================
# METRICS

def get_metrics():
    train_loss = tf.keras.metrics.Mean(name='loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='accuracy')
    return [train_loss,train_accuracy]
def single_metric_string(metric):
    return '{} {:.4f}'.format(metric.name,metric.result())
def metrics_strings(metrics):
    return ' '.join([single_metric_string(metric) for metric in metrics])
#==========================================================================================
# TRAIN
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]

@tf.function(input_signature=train_step_signature)
def train_step(inp,tar):
    global transformer,optimizer,metrics
    enc_padding_mask,combined_mask, dec_padding_mask = create_masks(inp,tar)
    faker = tf.zeros_like(tar)
    with tf.GradientTape() as tape:
      predictions, _ = transformer(inp,faker, 
                                  True,
                                  enc_padding_mask,
                                  combined_mask, 
                                  dec_padding_mask)
      loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    #print(transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    metrics[0](loss)
    metrics[1](tar,predictions)
    #train_loss(loss)
    #train_accuracy(tar, predictions)

def train(train_dataset):
    global c,chkpt_manager,metrics
    metrics = get_metrics()
    for epoch in range(c.train_kwargs['epochs']):
      start = time.time()
      for metric in metrics:
          metric.reset_states()
      
      for batch, seqs in enumerate(train_dataset):
        inp,tar = seqs
        #print(inp,tar)
        train_step(inp,tar)
        
        #if (batch % 50) == 0:
        #  print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
        #      epoch + 1, batch, train_loss.result(), train_accuracy.result()))
          
      if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
        
      print ('Epoch {} {}'.format(epoch + 1, metrics_strings(metrics)))
      print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
#==========================================================================================

# MAIN
def main():
    train_dataset=create_tf_dataset()
    load_model()
    get_optimizer()
    train(train_dataset)
main()