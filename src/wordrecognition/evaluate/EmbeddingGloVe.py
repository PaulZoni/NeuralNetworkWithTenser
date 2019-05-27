from keras.preprocessing.text import one_hot
from numpy import asarray, zeros
from keras.preprocessing.sequence import pad_sequences
from src.wordrecognition.nn.EmbeddingWithFullyConnected import EmbeddingWithFullyConnected
from keras.preprocessing.text import Tokenizer

docs = ['Well done!', 'Good work',
        'Great effort', 'nice work', 'Excellent!', 'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
max_length = 4
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
vec_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/glove.6B/glove.6B.100d.txt'
file = open(vec_path, mode='rt', encoding='utf-8')

embeddings_index = dict()
for line in file:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()
print('Loaded %s word vectors.' % len(embeddings_index))

print('vocab size: ' + str(vocab_size))
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = EmbeddingWithFullyConnected.build(vocab_size=vocab_size, input_length=max_length, out=100, weight_matrix=embedding_matrix, train=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(x=padded_docs, y=labels, epochs=50, verbose=0)
loss, accuracy = model.evaluate(x=padded_docs, y=labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
