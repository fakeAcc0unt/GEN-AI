
# import gensim
from gensim.models import Word2Vec
import numpy as np
import faiss

corpus = [
    'machine learning is fascinating',
    'python is a great programming language '
    'java and python are popular programing languages'
    'AI is transforming industries',
    'data science uses machine learning',
    'artificial intelligence is the future'
]

tokens = [i.lower().split() for i in corpus]
# print(tokens)

model = Word2Vec(sentences=tokens,vector_size=30,window=2,min_count=1,sg=1)


print(f'Vector for learning: {model.wv["learning"]}')
print(f'Most similar to learning: {model.wv.most_similar("learning")}')

# -----------------------------------------------------------------------------------------
# Faiss

words = list(model.wv.index_to_key)
print(words)
vectors = np.array([model.wv[i] for i in words]).astype('float32')

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

def find_similar(word):
    if word not in model.wv:
        return f'{word } not in vocabulary'

    vector = np.array([model.wv[word]]).astype('float32')
    distance,indices = index.search(vector,4)

    print(distance[0])
    print(indices[0])

    results = [( words[j],float(distance[0][i]) ) for i,j in enumerate(indices[0])]

    return results
print(find_similar('java'))


# -----------------------------------------------------------------------------------------
# GloVe

import gensim.downloader as gd

glove = gd.load('glove-wiki-gigaword-50')

print(f'Vector for cow: {glove["cow"]}')
print(f'Similarity between cow and buffalo {glove.similarity("cow","buffalo")}')