from src.wordrecognition.textpreparing import Tools
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense


def define_model(X):
    model = Sequential()
    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


sequences_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/char_sequences.txt'
mapping_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/res/mapping.pkl'
modal_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/model/model.h5'
raw_text = Tools.load_doc(file_path=sequences_path)
lines = raw_text.split('\n')
chars = sorted(list(set(raw_text)))
mapping = dict((char, i) for i, char in enumerate(chars))

sequences = list()
for line in lines:
    encode_sq = [mapping[char] for char in line]
    sequences.append(encode_sq)

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = array(sequences)
X, y = sequences[:, : -1], sequences[:, -1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

model = define_model(X)
model.fit(X, y, epochs=100, verbose=2)

model.save(modal_path)
dump(mapping, open(mapping_path, 'wb'))

