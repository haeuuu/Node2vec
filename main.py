from node2vec import Node2Vec
from skipgram import SkipGram

import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# examples
G = nx.karate_club_graph()
G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})


# node2vec random walks
node2vec = Node2Vec(G,
                    p = 1,
                    q = 0.0001,
                    min_edges = 1,
                    dim = 50,
                    walk_length = 10,
                    iterations = 2000,
                    num_workers = 5)
node2vec.generate_walks()


# skipgram
# NOTE : node2vec.fit() -> train gensim.word2vec
model = SkipGram(data = node2vec.walks,
                    hidden_size = 50,
                    negative_sample_size = 5,
                    padding_idx = -1,
                    max_vocab_size = -1,
                    max_len = -1,
                    min_counts = 0,
                    window_size = 5,
                    sampling = True,
                    subsampling_threshold = 1e-2)

num_workers = 8 if torch.cuda.is_available() else 0
model.fit(iterations = 10, batch_size = 512, num_workers = num_workers, shuffle = True)


# ref : https://frhyme.github.io/machine-learning/node2vec_lib/
node_features = model.features.to('cpu')
kmeans = KMeans(n_clusters = 5).fit(node_features)

for n, label in zip(model.vocab.index2entity.values(), kmeans.labels_):
    G.nodes[n]['kmeans-cluster'] = label

plt.figure(figsize=(10, 6))
nx.draw_kamada_kawai(G,
                     node_color = [n[1]['kmeans-cluster'] for n in G.nodes(data=True)],
                     cmap = plt.cm.rainbow,
                     with_labels = True)
plt.show()
