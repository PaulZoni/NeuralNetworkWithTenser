from src.wordrecognition.textpreparing import Tools

in_file_name = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/republic_clean.txt'
doc = Tools.load_doc(file_path=in_file_name)
tokens = Tools.clean_doc_v3(doc=doc)
print([tokens[: 200]])
print('Total tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
