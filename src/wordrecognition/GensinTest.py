from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/savemodal/word2vec_modal')
print(model.wv.most_similar(positive=['woman', 'king'], topn=5))
