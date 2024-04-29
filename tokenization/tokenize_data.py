from keras.preprocessing.text import Tokenizer
import numpy as np

from keras.utils.data_utils import pad_sequences

from training_configurations import training_configurations

max_words_in_dict = training_configurations['max_words_in_dict']
max_words_per_sample = training_configurations['max_words_per_sample']

program_specs = training_configurations['program_specs']

num_of_objects = program_specs['num_of_objects']

def tokenize_data(train_data):

    train_sequences = []
    train_tokenizers_dict = {}

    for ind, text in enumerate(train_data):
        print('text', ind, 'tokenization:')

        ### create tokenizer for text ###
        tokenizer = Tokenizer(num_words = max_words_in_dict)

        all_texts_list = text.flatten().tolist()

        tokenizer.fit_on_texts(all_texts_list)

        word_index = tokenizer.word_index
        print('Found', len(word_index), 'unique words in train data', ind)

        ### create sequences based on tokenizer dictionary ###
        sequences = []
        for object_i in range(num_of_objects):
            txt = text[:, object_i].tolist()
            seq = tokenizer.texts_to_sequences(txt)
            seq = pad_sequences(seq, maxlen = max_words_per_sample)                   # (samples, max_words_per_sample)
            seq = np.expand_dims(seq, axis=1)
            sequences.append(seq)

        sequences = np.concatenate(sequences, axis=1)
            
        print('Shape of data tensor', ind, 'after tokenization:', sequences.shape)                          #(samples,)

        train_sequences.append(sequences)
        train_tokenizers_dict[str(ind)] = tokenizer

    return train_sequences, train_tokenizers_dict






