import matplotlib.pyplot as plt
import spacy

from word2vec import Word2Vec


def plot_kmwq(word2vec_2d):
    kmwq = ['king', 'man', 'queen', 'woman']
    vecs = [word2vec_2d[word] for word in kmwq]
    colors = ['blue', 'green', 'red', 'orange']
    for color, word, (x, y) in zip(colors, kmwq, vecs):
        plt.scatter(x, y, label=word, c=color)

    king_v, man_v, queen_v, woman_v = vecs
    pairs = [(king_v, man_v), (queen_v, woman_v)]
    for ((x1, y1), (x2, y2)), color in zip(pairs, ['blue', 'red']):
        plt.plot([x1, x2], [y1, y2], c=color)

    plt.legend()
    plt.show()


def main():
    nlp_vecs = spacy.load('en_vectors_web_lg')
    word2vec = {lexeme.text: lexeme.vector for lexeme in nlp_vecs.vocab}
    word2idx = {lexeme.text: i for i, lexeme in enumerate(nlp_vecs.vocab)}
    word2vec_2d = Word2Vec(word2vec, word2idx, d=2)
    plot_kmwq(word2vec_2d)


if __name__ == '__main__':
    main()
