from keras.preprocessing.text import Tokenizer
import numpy as np

from keras.utils.data_utils import pad_sequences

from user_scripts.training_configurations import training_configurations

max_words_in_dict = training_configurations['max_words_in_dict']
max_words_per_sample = training_configurations['max_words_per_sample']

neurasp_conf = training_configurations['neurasp_conf']
classifiers_conf = neurasp_conf['classifiers_conf']
# sort classifiers_conf dictionary
classifiers_conf = dict(sorted(classifiers_conf.items()))

objects_of_classifier = {i: sorted(classifiers_conf[i]['objects']) for i in classifiers_conf}

def tokenize_data(train_data):

    train_sequences = {}
    train_tokenizers_dict = {}

    for ind_classifier in train_data:
        train_sequences_of_classifier = []
        for ind, text in enumerate(train_data[ind_classifier]):
            print('classifier', ind_classifier, 'text', ind, 'tokenization:')

            ### create tokenizer for text ###
            tokenizer = Tokenizer(num_words = max_words_in_dict)

            all_texts_list = text.flatten().tolist()

            tokenizer.fit_on_texts(all_texts_list)

            word_index = tokenizer.word_index
            print('Found', len(word_index), 'unique words in train data', ind, 'of classifier', ind_classifier)

            ### create sequences based on tokenizer dictionary ###
            sequences = []
            for object_i in range(len(objects_of_classifier[ind_classifier])):
                txt = text[:, object_i].tolist()
                seq = tokenizer.texts_to_sequences(txt)
                seq = pad_sequences(seq, maxlen = max_words_per_sample)                   # (samples, max_words_per_sample)
                seq = np.expand_dims(seq, axis=1)
                sequences.append(seq)

            sequences = np.concatenate(sequences, axis=1)
                
            print('Shape of data tensor', ind, 'of classifier', ind_classifier, 'after tokenization:', sequences.shape)                          #(samples,)

            train_sequences_of_classifier.append(sequences)
            train_tokenizers_dict[str(ind)+'_classifier_'+str(ind_classifier)] = tokenizer
        train_sequences[ind_classifier] = train_sequences_of_classifier

    return train_sequences, train_tokenizers_dict






