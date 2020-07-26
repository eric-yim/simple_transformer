#COMMON PARAMETERS
class Config:
    def __init__(self,max_num,pe_input=100):
        self.model_kwargs= {
            'num_layers':4,
            'd_model':32,
            'dff':64,
            'num_heads':4,
            'input_vocab_size':max_num+1,
            'target_vocab_size':max_num+1,
            'pe_input':pe_input,
            'pe_target':2,
            'rate':0.1 #(Dropout)
            }
        self.checkpoint_path="./checkpoints/"
        self.custom_lr = False
        self.train_kwargs={
            'epochs':200,
            'buffer_size':100,
            'batch_size':8
        }
        self.data_path= 'fake_datalist_0.csv'