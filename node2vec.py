import time
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import defaultdict
import multiprocessing as mp
from gensim.models import Word2Vec

class Node2Vec:
    WEIGHT_KEY = 'weight'
    FIRST_STEP_KEY = 'first_step'
    TRANSITION_PROB_KEY = 'prob'
    NEIGHBORS_KEY = 'neighbor'

    def __init__(self, graph, p=1, q=1, min_edges=1, dim=128, walk_length=10, iterations=10, num_workers=5):
        self.p, self.q = p, q
        self.min_edges = min_edges
        self.dim = dim
        self.walk_length = walk_length
        self.iterations = iterations
        self.num_workers = num_workers

        self.node2vec_rw_graph = self.precompute_transition_prob(graph)

    def precompute_transition_prob(self, graph):
        """convert graph to node2vec_graph
        Input:
            graph : nx.graph
        Return:
            node2vec_graph : defaultdict
            node2vec_graph[src][dst][NEIGHBORS_KEY] : list of neighbors 
            node2vec_graph[src][dst][WEIGHT_KEY] : np.array of transition weights
        """
        node2vec_graph = defaultdict(dict)

        for src in graph.nodes():

            # for curr ~ 안으로 넣을 수 있음.
            frist_step_weights = [graph[src][dst].get(self.WEIGHT_KEY, 1) for dst in graph[src].keys()]
            neighbor_keys = list(graph[src].keys())

            if len(neighbor_keys) < self.min_edges:
                continue

            node2vec_graph[src][self.FIRST_STEP_KEY] = {
                self.TRANSITION_PROB_KEY: noramlize(frist_step_weights),
                self.NEIGHBORS_KEY: neighbor_keys
            }

            for curr in graph[src]:

                transition_weights, neighbor_keys = [], []

                for dst in graph[curr].keys():
                    weight = graph[curr][dst].get(self.WEIGHT_KEY, 1)

                    if src == dst:  # distance = 0
                        weight *= 1 / self.p
                    elif graph[src].get(dst) is not None:  # distance = 1
                        weight *= 1
                    else:  # distance = 2
                        weight *= 1 / self.q

                    neighbor_keys.append(dst)
                    transition_weights.append(weight)

                normalized_transition_weights = noramlize(transition_weights)

                node2vec_graph[src][curr] = {
                    self.TRANSITION_PROB_KEY: normalized_transition_weights,
                    self.NEIGHBORS_KEY: neighbor_keys
                }

        return node2vec_graph

    def _get_next_node(self, neighbors_with_probs):
        next_node = np.random.choice(neighbors_with_probs[self.NEIGHBORS_KEY],
                                     size=1,
                                     p=neighbors_with_probs[self.TRANSITION_PROB_KEY])

        return next_node[0]

    def _random_walk(self, inQueue, outQueue):

        while True:
            src = inQueue.get()

            init_probs = self.node2vec_rw_graph[src][self.FIRST_STEP_KEY]
            
            for i in range(self.iterations):
                first_step = self._get_next_node(init_probs)

                walks = [src, first_step]
                for _ in range(self.walk_length - 2):
                    prev, curr = walks[-2], walks[-1]
                    next_node = self._get_next_node(self.node2vec_rw_graph[prev][curr])
                    walks.append(next_node)

                outQueue.put(walks)

    def _start(self):
        self.inQueue = mp.Queue()
        self.outQueue = mp.Queue()

        for src in self.node2vec_rw_graph.keys():
            self.inQueue.put(src)

        self.processes = [mp.Process(target = self._random_walk, args = (self.inQueue, self.outQueue)) \
            for _ in range(self.num_workers)]

        for p in self.processes:
            p.start()

    def _collect(self):
        all_walks = []
        num_walks = len(self.node2vec_rw_graph) * self.iterations
        pbar = tqdm(total = num_walks)

        while True:
            all_walks.append(self.outQueue.get())
            pbar.update(1)
            if len(all_walks) >= num_walks:
                break
        pbar.close()

        return [list(map(str, w)) for w in all_walks]

    def _terminate(self):
        for p in self.processes:
            p.terminate()
            p.join() 

    def generate_walks(self):
        self._start()
        self.walks = self._collect()
        self._terminate()

    def fit(self, **w2v_params):
        """gensim w2v default settings
        size = 100
        window = 5
        min_count = 5
        workers = 3
        sg = 0
        negative = 5
        iter = 5
        """

        print('Generate walks ...')
        self.generate_walks()

        print('Train W2V ...')
        return Word2Vec(self.walks, vector_size = self.dim, sg = 1, min_count = 0, **w2v_params)

def noramlize(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    return array / array.sum()

if __name__ == '__main__':
    # examples
    G = nx.karate_club_graph()
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    model = Node2Vec(G,
                     p = 1,
                     q = 0.0001,
                     min_edges = 1,
                     dim = 50,
                     walk_length = 10,
                     iterations = 2000,
                     num_workers = 5)
    w2v = model.fit()

    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # ref : https://frhyme.github.io/machine-learning/node2vec_lib/
    node_features = w2v.wv.vectors
    kmeans = KMeans(n_clusters = 5).fit(node_features)

    # index2entity has been replaced by index_to_key (gensim 4.0.0)
    for n, label in zip(w2v.wv.index_to_key, kmeans.labels_):
        G.nodes[n]['cluster'] = label

    plt.figure(figsize=(10, 6))
    nx.draw_kamada_kawai(G,
                         node_color = [n[1]['cluster'] for n in G.nodes(data=True)],
                         cmap = plt.cm.rainbow,
                         with_labels = True)
    plt.show()
