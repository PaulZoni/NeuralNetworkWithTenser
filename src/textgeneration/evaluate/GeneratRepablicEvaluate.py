from src.wordrecognition.textpreparing import Tools
from pickle import load
from keras.models import load_model
from keras.models import Sequential
from random import randint
from src.textgeneration.prepare.GnerateTextTools import generate_sequence


in_filename = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/republic_sequences.txt'
model_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/model/model_plato.h5'
tokenizer_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/res/tokenizer.pkl'
doc = Tools.load_doc(file_path=in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

model: Sequential = load_model(filepath=model_path)
tokenizer = load(open(file=tokenizer_path, mode='rb'))
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')
generated = generate_sequence(model=model, tokenizer=tokenizer, max_length=seq_length, seed_text=seed_text, n_word=50)
print(generated)
