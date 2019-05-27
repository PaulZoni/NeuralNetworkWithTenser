from src.wordrecognition.textpreparing import Tools
from keras.models import load_model, Model

path_neg = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/neg'
path_pos = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/pos'
vocab_save_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/vocab.txt'
save_modal = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/savemodal/ModelOneCnnLayer.h5'

vocab = Tools.load_doc(file_path=vocab_save_path)
vocab = set(vocab.split())
train_docs, y_train = Tools.load_clean_dataset(vocab=vocab, path_pos=path_pos, path_neg=path_neg, is_train=True)
test_docs, y_test = Tools.load_clean_dataset(vocab=vocab, path_pos=path_pos, path_neg=path_neg, is_train=False)

tokenizer = Tools.create_tokenizer(train_docs)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)

X_train = Tools.encode_docs(tokenizer=tokenizer, max_length=max_length, docs=train_docs)
X_test = Tools.encode_docs(tokenizer=tokenizer, max_length=max_length, docs=test_docs)

model: Model = load_model(save_modal)
_, acc = model.evaluate(x=X_train, y=y_train, verbose=0)
print('Train Accuracy: %.2f' % (acc*100))
_, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.2f' % (acc*100))

text_positive = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = Tools.predict_sentiment_v2(review=text_positive, vocab=vocab, tokenizer=tokenizer, model=model, max_length=max_length)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text_positive, sentiment, percent*100))
text_negative = 'This is a bad movie. Do not watch it. It sucks.'
percent, sentiment = Tools.predict_sentiment_v2(review=text_negative, vocab=vocab, tokenizer=tokenizer, model=model, max_length=max_length)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text_negative, sentiment, percent*100))
