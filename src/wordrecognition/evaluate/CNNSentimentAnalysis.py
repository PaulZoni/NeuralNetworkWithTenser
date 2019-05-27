import matplotlib.pyplot as pyplot
from src.wordrecognition.textpreparing import Tools
from src.wordrecognition.nn.ModelOneCnnLayer import ModelOneCnnLayer

path_neg = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/neg'
path_pos = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/pos'
vocab_save_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/vocab.txt'
save_modal = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/savemodal/ModelOneCnnLayer.h5'

vocab = Tools.load_doc(file_path=vocab_save_path)
vocab = set(vocab.split())
train_docs, y_train = Tools.load_clean_dataset(vocab=vocab, path_neg=path_neg, path_pos=path_pos, is_train=True)
print(len(train_docs))
tokenizer = Tools.create_tokenizer(lines=train_docs)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)

X_train = Tools.encode_docs(tokenizer=tokenizer, max_length=max_length, docs=train_docs)
model = ModelOneCnnLayer.build(vocab_size=vocab_size, max_length=max_length)
history = model.fit(x=X_train, y=y_train, epochs=10, verbose=2)
model.save(save_modal)

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['loss'])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['acc', 'loss'], loc='upper left')
pyplot.show()
pyplot.savefig('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/plot/RnnForWord_plot.png')


