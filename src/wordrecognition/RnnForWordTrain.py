from keras.preprocessing import sequence
from src.wordrecognition.nn.RnnForWord import RnnForWord
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import pandas
from keras.optimizers import SGD

embedding_vector_length = 100

df = pandas.DataFrame()
df = pandas.read_csv('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/imdb_master.csv',
                     encoding='ISO-8859-1')
df.head(3)
X_train = df.loc[12000: 13000, 'review']
y_train = df.loc[12000: 13000, 'label']

tk = Tokenizer()
total_review = X_train + y_train
tk.fit_on_texts(total_review)
print(str(df.size))

length_max = max([len(s.split()) for s in total_review])
vocabulary_size = len(tk.word_index) + 1

for index, value in y_train.items():
    if value == 'neg':
        y_train[index] = 0
    else:
        y_train[index] = 1

print(len(y_train))
print(len(X_train))
print(str(y_train))
X_train_tokens = tk.texts_to_sequences(X_train)

X_train_bad = sequence.pad_sequences(X_train_tokens, maxlen=length_max, padding='post')

optimizer = SGD(lr=0.08, decay=1e-6, momentum=0.9, nesterov=True)
model = RnnForWord().build(vocab_size=vocabulary_size,
                           max_review_length=length_max, embedding_vector_length=length_max)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train_bad, y_train, epochs=3, batch_size=32)

index_list = tk.texts_to_sequences([X_train[12000],
                                    X_train[13000]
                                    ])
test = sequence.pad_sequences(index_list, maxlen=length_max)

answer = model.predict(x=test)
print('predict: ' + str(answer))

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'loss'], loc='upper left')
plt.savefig('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/plot/RnnForWord_plot.png')
plt.show()
