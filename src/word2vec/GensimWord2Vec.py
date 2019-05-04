from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Word2Vec

wiki = WikiCorpus('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/word2vec/res/ruwiki-20190120-pages-articles-multistream1.xml-p4p204179.bz2', dictionary=False)
print('bigram')
bigram = Phrases(wiki.get_texts())
print('bigram_transformer')
bigram_transformer = Phraser(bigram)


def text_generator_bigram():
    for text in wiki.get_texts():
        yield bigram_transformer[[word for word in text]]


trigram = Phrases(text_generator_bigram())
print('trigram')
trigram_transformer = Phraser(trigram)
print('trigram_transformer')


def text_generator_trigram():
    for text in wiki.get_texts():
        yield trigram_transformer[bigram_transformer[[word for word in text]]]


print('model create')
model = Word2Vec(size=100, window=7, min_count=10, workers=10, iter=1, min_alpha=0.025)
print('build_vocab')
model.build_vocab(text_generator_trigram())
print('train model')
model.train(text_generator_trigram(), epochs=1, total_examples=model.corpus_count)
print('save')
model.save('/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/word2vec/savemodal/word2vec_modal')
