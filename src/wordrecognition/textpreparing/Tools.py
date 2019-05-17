import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import HashingVectorizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.preprocessing.text import Tokenizer # define 5 documents


'''filename = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()'''


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



