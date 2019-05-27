from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from src.wordrecognition.nn.EmbeddingWithFullyConnected import EmbeddingWithFullyConnected

docs = ['Well done!', 'Good work',
        'Great effort', 'nice work', 'Excellent!', 'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
vocab_size = 50
max_length = 4
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

model = EmbeddingWithFullyConnected.build(vocab_size=vocab_size, input_length=max_length, out=8, weight_matrix=None)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(x=padded_docs, y=labels, epochs=50, verbose=0)
loss, accuracy = model.evaluate(x=padded_docs, y=labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
