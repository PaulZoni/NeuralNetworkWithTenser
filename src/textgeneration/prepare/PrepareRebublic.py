from src.wordrecognition.textpreparing import Tools

in_file_name = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/republic_clean.txt'
doc = Tools.load_doc(file_path=in_file_name)
tokens = Tools.clean_doc_v3(doc=doc)
print([tokens[: 200]])
print('Total tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i - length: i]
    line = ' '.join(seq)
    sequences.append(line)

print('Total sequences: %d' % len(sequences))
out_file_name = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/republic_sequences.txt'
Tools.save_list(lines=sequences, file_name=out_file_name)


