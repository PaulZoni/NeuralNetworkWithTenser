from src.textgeneration.rnn.ModelAsFunc import define_model_with_two_LSTM as define_model
from src.wordrecognition.textpreparing import Tools
from keras.preprocessing.text import Tokenizer
from numpy import array
from pickle import dump
from keras.utils import to_categorical

in_filename = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/republic_sequences.txt'
doc = Tools.load_doc(file_path=in_filename)

lines = doc.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=lines)
sequences = tokenizer.texts_to_sequences(texts=lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y=y, num_classes=vocab_size)
seq_length = X.shape[1]
print(seq_length)

model = define_model(vocab_size=vocab_size, output=50, seq_length=seq_length)
history = model.fit(x=X, y=y, batch_size=128, epochs=10)
model.save('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/model/model_plato.h5')
dump(tokenizer, open('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/res/tokenizer.pkl', 'wb'))
Tools.plot_history(history=history)
