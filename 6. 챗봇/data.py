from preprocess import *
import json
import numpy as np

PATH = 'data_in/ChatbotData.csv'
VOCAB_PATH = 'data_in/vocabulary.txt'

inputs, inputs_val, outputs, outputs_val = load_data(PATH)

char2idx, idx2char, vocab_size = load_vocabulary(PATH, VOCAB_PATH)

print("Vocabulary  completed.")

index_inputs, input_seq_len = enc_processing(inputs, char2idx)
index_outputs, output_seq_len = dec_output_processing(outputs, char2idx)
index_targets = dec_target_processing(outputs, char2idx)
index_inputs_val, input_val_seq_len = enc_processing(inputs_val, char2idx)
index_output_val, output_val_seq_len = dec_output_processing(outputs_val, char2idx)
index_targets_val = dec_target_processing(outputs_val, char2idx)

print("Preprocessing completed.")

data_configs = {}
data_configs['char2idx'] = char2idx
data_configs['idx2char'] = idx2char
data_configs['vocab_size'] = vocab_size
data_configs['pad_symbol'] = PAD
data_configs['std_symbol'] = START
data_configs['end_symbol'] = END
data_configs['unk_symbol'] = UNK


DATA_IN_PATH = './data_in/'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
VAL_INPUTS = 'val_inputs.npy'
VAL_OUTPUTS = 'val_outputs.npy'
VAL_TARGETS = 'val_targets.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH + TRAIN_INPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'wb'), index_outputs)
np.save(open(DATA_IN_PATH + TRAIN_TARGETS , 'wb'), index_targets)
np.save(open(DATA_IN_PATH + VAL_INPUTS, 'wb'), index_inputs_val)
np.save(open(DATA_IN_PATH + VAL_OUTPUTS , 'wb'), index_output_val)
np.save(open(DATA_IN_PATH + VAL_TARGETS , 'wb'), index_targets_val)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w', encoding='utf-8'))
print("Data is prepared.")
