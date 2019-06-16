from src.textgeneration.rnn.ModelAsFunc import define_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
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


data = """ Jack and Jill went up the hill\n
    To fetch a pail of water\n
    Jack fell down and broke his crown\n
    And Jill came tumbling after\n """

tokenizer: Tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
vocab_size = len(tokenizer.word_counts) + 1
print('Vocabulary size: %x' % vocab_size)

sequences = list()
for line in data.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
print('Total Sequence: %d' % len(sequences))

max_len = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences=sequences, maxlen=max_len)
print('Max Sequence Length: %d' % max_len)

sequences = np.array(sequences)

X, y = sequences[:, : -1], sequences[:, -1]
y = to_categorical(y=y, num_classes=vocab_size)

model = define_model(vocab_size=vocab_size, input_length=max_len - 1)
model.fit(x=X, y=y, epochs=500, verbose=2)

print(generate_sequence(model=model, tokenizer=tokenizer, max_length=max_len-1, seed_text='Jack', n_word=4))
print(generate_sequence(model=model, tokenizer=tokenizer, max_length=max_len-1, seed_text='Jill', n_word=4))
