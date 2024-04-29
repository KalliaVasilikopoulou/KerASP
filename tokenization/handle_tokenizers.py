import pickle


def save_tokenizers(tokenizer_dict):
    for tokenizer_name, tokenizer in tokenizer_dict.items():
        with open('train_data/train_tokenizer_' + tokenizer_name + '.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved ' + tokenizer_name)

    return None


def load_tokenizers(tokenizer_names):   # input = list of strings
    tokenizer_dict = {}
    for tokenizer_name in tokenizer_names:
        with open('train_data/train_tokenizer_' + tokenizer_name + '.pickle', 'rb') as handle:
            tokenizer_dict[tokenizer_name] = pickle.load(handle)
        print('loaded ' + tokenizer_name)

    return tokenizer_dict
