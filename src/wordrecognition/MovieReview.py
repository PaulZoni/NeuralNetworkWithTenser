import src.wordrecognition.textpreparing.Tools as tools
from src.wordrecognition.nn.FullyConnected import FullyConnected


file_pos_dir = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/pos/'
file_neg_dir = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/neg/'
vocab_directory = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/vocab.txt'

vocab = tools.load_doc(vocab_directory)
vocab = vocab.split()
vocab = set(vocab)

train_docs, y_train = tools.load_clean_dataset(vocab=vocab)
test_docs, y_test = tools.load_clean_dataset(vocab=vocab)

tokenizer = tools.create_tokenizer(train_docs)
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='binary')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='binary')
n_words = Xtrain.shape[1]

model = FullyConnected.build(input_size=n_words)
model.fit(x=Xtrain, y=y_train, epochs=10)

text = 'Best movie ever! It was great, I recommend it.'
percent, sentiment = tools.predict_sentiment(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent * 100))

text = 'This is a bad movie.'
percent, sentiment = tools.predict_sentiment(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
