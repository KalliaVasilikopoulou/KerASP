import numpy as np

import regex as re

from string import punctuation
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

lem = SnowballStemmer('english')

stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)

def clean_text(line):
    line = str(line)
    line = re.sub(r'[^a-zA-Z ]','',line)
    line = re.sub(r'[ ]+',' ',line)
    line = line.lower()
    line_list = line.split() # split to individual words (returns list)
    line_list = [lem.stem(word) for word in line_list if not word in stuff_to_be_removed]
    line = ' '.join(line_list)
    
    return np.array(line)

clean_text_vec = np.vectorize(clean_text)
