from keras.datasets import imdb
from keras.preprocessing import sequence
from src.word2vec.nn.RnnForWord import RnnForWord
import matplotlib.pyplot as plt

top_words = 5000
max_review_length = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32

model = RnnForWord().build(top_words=top_words,
                           max_review_length=max_review_length, embedding_vector_length=embedding_vector_length)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=3, batch_size=64,)
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy : %.4f ' % scores[1])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='upper left')
plt.show()
plt.savefig('')

