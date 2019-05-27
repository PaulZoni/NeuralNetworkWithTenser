from gensim.models import Word2Vec, KeyedVectors
from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.scripts.glove2word2vec import glove2word2vec


sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'], ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]


def plot_vocab():
    model = Word2Vec(sentences=sentences, min_count=1)
    X = model.wv.__getitem__(model.wv.vocab)
    pca = PCA(n_components=2)
    result = pca.fit_transform(X=X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)

    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def test_CBOW(file_name, binary=True):
    model = KeyedVectors.load_word2vec_format(fname=file_name, binary=binary)
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
    print(result)


glove_input_file = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/res/glove.6B/glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file=glove_input_file, word2vec_output_file=word2vec_output_file)
filename = word2vec_output_file
test_CBOW(file_name=filename, binary=False)
