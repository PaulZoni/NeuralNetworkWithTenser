from keras.preprocessing.text import Tokenizer
from src.captions.ClearDiscriptions import load_clean_descriptions
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array, shape
from os import listdir
from os import path
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_clean_descriptions(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        descriptions[image_id] = ' '.join(image_desc)
    return descriptions


def create_tokenizer(descriptions):
  lines = list(descriptions.values())
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer


def load_photo(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)[0]
    image_id = path.basename(filename).split('.')[0]
    return image, image_id


def create_sequences(tokenizer, max_length, desc, image):
    y = []
    XSeq = []
    Ximages = []

    vocab_size = len(tokenizer.word_index) + 1
    seq = tokenizer.texts_to_sequences([desc])[0]
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        Ximages.append(image)
        XSeq.append(in_seq)
        y.append(out_seq)
    Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
    return [Ximages, XSeq, y]


def data_generator(descriptions, tokenizer, max_length):
    directory = '/Users/p.polyakov/Documents/data_set/Flickr8k/Flicker8k_Dataset/'
    while 1:
        for name in listdir(directory):
            filename = path.join(directory, name)
            image, image_id = load_photo(filename)
            desc = descriptions[image_id]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
            yield [[in_img, in_seq], out_word]



descriptions = load_clean_descriptions('/Users/p.polyakov/Documents/data_set/descriptions.txt')
tokenizer = create_tokenizer(descriptions)
# pad to fixed length
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)
# test the data generator
generator = data_generator(descriptions=descriptions, tokenizer=tokenizer, max_length=max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)












