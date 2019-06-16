from keras.preprocessing.sequence import pad_sequences
from src.wordrecognition.textpreparing import Tools
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model
from pickle import load
from numpy import array

sequences_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/char_sequences.txt'
mapping_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/res/mapping.pkl'
modal_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/model/model.h5'


def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text

    for _ in range(n_chars):

        encoded = [mapping[char] for char in in_text]

        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = array(encoded)
        yhat = model.predict_classes(encoded, verbose=0)

        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break

        in_text += out_char
    return in_text


model = load_model(modal_path)
mapping = load(open(mapping_path, 'rb'))

print(generate_seq(model, mapping, 10, 'Sing a son', 20))
print(generate_seq(model, mapping, 10, 'king was i', 20))
print(generate_seq(model, mapping, 10, 'hello worl', 20))
