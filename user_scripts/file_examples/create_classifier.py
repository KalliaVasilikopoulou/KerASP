import keras
from keras import layers

from training_configurations import training_configurations

max_words_in_dict = training_configurations['max_words_in_dict']
max_words_per_sample = training_configurations['max_words_per_sample']
embedding_dim = training_configurations['embedding_dim']


classifier_input = [keras.Input(shape=(max_words_per_sample,)) for _ in range(2)]   # [(None,max_words_per_sample),(None,max_words_per_sample)]

x = [layers.Embedding(max_words_in_dict, embedding_dim, input_length=max_words_per_sample)(i) for i in classifier_input]

x = [layers.Bidirectional(layers.LSTM(32))(i) for i in x]

x = layers.Concatenate()(x)

x = layers.Dense(32, activation='relu')(x)

classifier_output = layers.Dense(10, activation = 'softmax', name='article_topics_probs')(x)        # we have 10 topics, so 10 classes

classifier = keras.models.Model(classifier_input, classifier_output, name='classifier')
