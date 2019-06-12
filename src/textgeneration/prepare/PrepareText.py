from src.wordrecognition.textpreparing import Tools

text_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/rhyme.txt'

sequences_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/char_sequences.txt'
raw_text = Tools.load_doc(file_path=text_path)
print(raw_text, '\n')

tokens = raw_text.split()
raw_text = ' '.join(tokens)

length = 10
sequences = list()

for i in range(length, len(raw_text)):
    seq = raw_text[i - length: i + 1]
    sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
print(sequences)

Tools.save_list(lines=sequences, file_name=sequences_path)

