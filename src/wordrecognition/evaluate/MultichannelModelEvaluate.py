from src.wordrecognition.textpreparing import Tools
from src.wordrecognition.nn.MultichannelCnnModel import MultichannelCnnModel
import matplotlib.pyplot as pyplot

save_modal = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/savemodal/MultichannelCnnModel.h5'
vocab_save_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/vocab.txt'
vocab = Tools.load_doc(file_path=vocab_save_path)
vocab = set(vocab.split())

train_docs, ytrain = Tools.load_clean_dataset(is_train=True, vocab=vocab)
test_docs, ytest = Tools.load_clean_dataset(is_train=False, vocab=vocab)

tokenizer = Tools.create_tokenizer(lines=train_docs)
length = Tools.max_length(lines=train_docs)
print('Max document length: %d' % length)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

trainX = Tools.encode_docs(tokenizer=tokenizer, docs=train_docs, max_length=length)
testX = Tools.encode_docs(tokenizer=tokenizer, docs=test_docs, max_length=length)

model = MultichannelCnnModel.build(length=length, vocab_size=vocab_size)
history = model.fit(x=[trainX, trainX, trainX], y=ytrain, validation_data=([testX, testX, testX], ytest), epochs=10, batch_size=16)
model.save(save_modal)
Tools.plot_history(history=history)
