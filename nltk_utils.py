# Importing required libraries
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


# Tokenize the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Applying Stemming
def stem(word):
    return  stemmer.stem(word.lower())


# bag of words
def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]\
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [    0,     1,    0,      1,    0,      0,        0]
    :param tokenized_sentence:
    :param all_words:
    :return:
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    # loop over all_words
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
