from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from src.textgeneration.rnn.ModelAsFunc import define_model
from src.textgeneration.prepare.GnerateTextTools import generate_sequence

data = """ Jack and Jill went up the hill\n
    To fetch a pail of water\n
    Jack fell down and broke his crown\n
    And Jill came tumbling after\n """

tokenizer: Tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
vocab_size = len(tokenizer.word_counts) + 1
print('Vocabulary size: %x' % vocab_size)
encoded = tokenizer.texts_to_sequences([data])[0]

sequences = list()
for i in range(2, len(encoded)):
    sequence = encoded[i - 2: i + 1]
    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))
max_length = max([len(seq) for seq in sequences])
print('Max Sequence Length: %d' % max_length)
sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y=y, num_classes=vocab_size)

model = define_model(vocab_size=vocab_size, input_length=max_length - 1)
model.fit(x=X, y=y, epochs=500, verbose=2)

print(generate_sequence(model, tokenizer, max_length-1, 'Jack and', 5))
print(generate_sequence(model, tokenizer, max_length-1, 'And Jill', 3))
print(generate_sequence(model, tokenizer, max_length-1, 'fell down', 5))
print(generate_sequence(model, tokenizer, max_length-1, 'pail of', 5))

