from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np


def generate_sequence(model: Sequential, tokenizer: Tokenizer, max_length, seed_text, n_word):
    in_text = seed_text
    for _ in range(n_word):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded)
        encoded = pad_sequences(sequences=[encoded], maxlen=max_length)
        yhat = model.predict_classes(x=encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word

    return in_text
