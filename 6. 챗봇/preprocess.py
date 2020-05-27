from configs import DEFINES
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt

FILTERS = "([~.,!?':;)(])"
PAD = "<PADDING>"
START = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX, START_INDEX, END_INDEX, UNK_INDEX = 0, 1, 2, 3
MARKER = [PAD, START, END, UNK]

CHANGE_FILTER = re.compile(FILTERS)

def load_data(path=DEFINES['data_path']):
    data_df = pd.read_csv(path, header=0, encoding='utf-8')
    question, answer = list(data_df['Q']), list(data_df['A'])
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=.1, random_state=42)
    return train_input, eval_input, train_label, eval_label

def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in data:
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)
    return result_data

def load_vocabulary(path=DEFINES['data_path'], vocab_path=DEFINES['vocab_path'], tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    char2idx, idx2char = make_vocabulary(vocabulary_list)

    return char2idx, idx2char, len(char2idx)

def make_vocabulary(vocabulary_list):
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    return char2idx, idx2char

def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_index = []
    sequences_length = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])
        if len(sequence_index) > DEFINES['max_sequence_length']:
            sequence_index = sequence_index[:DEFINES['max_sequence_length']]
        sequences_length.append(len(sequence_index))
        sequence_index += (DEFINES['max_sequence_length'] - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)
    return np.asarray(sequences_input_index), sequences_length


def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[START]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        if len(sequence_index) > DEFINES['max_sequence_length']:
            sequence_index = sequence_index[:DEFINES['max_sequence_length']]
        sequences_length.append(len(sequence_index))
        sequence_index += (DEFINES['max_sequence_length'] - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)
    return np.asarray(sequences_output_index), sequences_length


def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    sequences_target_index = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        if len(sequence_index) >= DEFINES['max_sequence_length']:
            sequence_index = sequence_index[:DEFINES['max_sequence_length'] - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        sequence_index += (DEFINES['max_sequence_length'] - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)
    return np.asarray(sequences_target_index)




