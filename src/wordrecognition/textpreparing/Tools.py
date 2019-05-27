import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from pickle import dump
import matplotlib.pyplot as plt

file_pos_dir = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/pos/'
file_neg_dir = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/review_polarity/txt_sentoken/neg/'


def prepare_text(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    stripped = [re_punc.sub('', w) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words]


def word_steam(tokens):
    porter = PorterStemmer()
    return [porter.stem(word) for word in tokens]


def load_doc(file_path=None):
    file = open(file_path, 'r')
    text = file.read()
    file.close()
    return text


def process_docs_to_list(directory, vocab):
    list_doc = []
    for file in listdir(directory):
        if not file.endswith('.txt'):
            next
        path = directory + '/' + file
        doc = doc_to_line(file_name=path, vocab=vocab)
        list_doc.append(doc)
    return list_doc


def process_docs_with_test_split(directory, vocab, startswith='cv9', is_train=False):
    list_doc = []
    for file in listdir(directory):
        if is_train and file.startswith(startswith):
            continue
        if not is_train and not file.startswith(startswith):
            continue
        path = directory + '/' + file
        doc = doc_to_line(file_name=path, vocab=vocab)
        list_doc.append(doc)
    return list_doc


def clean_doc(doc: str):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def clean_doc_v2(doc, vocab):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(directory, vocab):
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            next
        path = directory + '/' + filename
        add_doc_to_vocab(filename=path, vocab=vocab)


def save_list(lines, file_name):
    data = '\n'.join(lines)
    file = open(file_name, 'w')
    file.write(data)
    file.close()


def doc_to_line(file_name, vocab):
    doc = load_doc(file_path=file_name)
    tokens = clean_doc(doc=doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def load_clean_dataset(vocab, path_neg=file_neg_dir, path_pos=file_pos_dir, is_train=False):
    neg = process_docs_with_test_split(vocab=vocab, directory=path_neg, is_train=is_train)
    pos = process_docs_with_test_split(vocab=vocab, directory=path_pos, is_train=is_train)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


def create_tokenizer(lines) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def predict_sentiment(review, vocab, tokenizer: Tokenizer, model: Model):
    tokens = clean_doc(doc=review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encoded, verbose=0)
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


def predict_sentiment_v2(review, vocab, tokenizer, max_length, model):
    line = clean_doc_v2(review, vocab)
    padded = encode_docs(tokenizer, max_length, [line])
    yhat = model.predict(padded, verbose=0)
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


def encode_docs(tokenizer: Tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    paded = pad_sequences(sequences=encoded, maxlen=max_length, padding='post')
    return paded


def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


def max_length(lines):
    return max([len(s.split()) for s in lines])


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

