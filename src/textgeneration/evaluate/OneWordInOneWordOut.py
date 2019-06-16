from src.textgeneration.rnn.ModelAsFunc import define_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np


def generate_seq(model: Sequential, tokenizer: Tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded)
        yhat = model.model.predict_classes(x=encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text, result = out_word, result + ' ' + out_word
    return result


data = """ Jack and Jill went up the hill\n
    To fetch a pail of water\n
    Jack fell down and broke his crown\n
    And Jill came tumbling after\n """

tokenizer: Tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_counts) + 1
print('Vocabulary size: %x' % vocab_size)

sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1: i+1]
    sequences.append(sequence)
print('Total sequence: %a' % len(sequences))
sequences = np.array(sequences)
X, y = sequences[:, 0], sequences[:, 1]

y = to_categorical(y=y, num_classes=vocab_size)
model = define_model(vocab_size, input_length=1)
model.fit(x=X, y=y, epochs=500, verbose=2)
print(generate_seq(model=model, tokenizer=tokenizer, seed_text='To', n_words=6))
