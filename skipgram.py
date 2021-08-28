import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

from vocab import Vocabulary

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SkipGramModel, self).__init__()
        self.U = nn.Embedding(vocab_size, hidden_size)
        self.V = nn.Embedding(vocab_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.U.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

    def forward(self, center_ids, context_neg_ids):
        center_embedding = self.U(center_ids)
        context_neg_embedding = self.V(context_neg_ids)  # (batch_size, max_len , hidden_size)

        center_embedding_t = center_embedding.transpose(1, 2)  # (batch_size, hidden_size, 1)
        dot_product = torch.bmm(context_neg_embedding, center_embedding_t)  # pairwise dot product

        return dot_product.squeeze(2)

class SkipGramDataset(Dataset):
    def __init__(self, center, context, negative_samples, max_len, padding_idx):
        self.center = center
        self.context = context
        self.negative_samples = negative_samples
        self.max_len = max_len
        self.padding_idx  = padding_idx

    def __len__(self):
        return len(self.center)

    def __getitem__(self, idx):
        """
            label : context (1) / negative or padding (0)
            mask : padding (0) / otherwise(1)
        """
        center = self.center[idx]
        positive = self.context[idx]
        negative = self.negative_samples[idx]

        padding_length = self.max_len - len(positive) - len(negative)

        pos_and_neg = positive + negative + [self.padding_idx] * padding_length
        label = [1.] * len(positive) + [0.] * (len(negative) + padding_length)
        mask = [1.] * (len(positive) + len(negative)) + [0.] * padding_length

        return [torch.tensor(center), torch.tensor(pos_and_neg), torch.tensor(label), torch.tensor(mask)]

class SkipGram(nn.Module):
    def __init__(self,
                 data,
                 hidden_size = 50,
                 negative_sample_size = 5,
                 padding_idx = -1,
                 max_vocab_size = -1,
                 max_len = -1,
                 min_counts = 0,
                 window_size = 5,
                 sampling = True,
                 subsampling_threshold = 1e-5):
        super(SkipGram, self).__init__()

        self.vocab = Vocabulary(data = data,
                                max_vocab_size = max_vocab_size,
                                min_counts = min_counts,
                                window_size = window_size,
                                negative_sample_size = negative_sample_size,
                                sampling = sampling,
                                subsampling_threshold = subsampling_threshold)

        self.vocab_size = self.vocab.vocab_size
        self.padding_idx = padding_idx
        if self.padding_idx < 0:
            self.padding_idx = self.vocab_size

        self.max_len = max_len
        if max_len < 0:
            self.max_len = 2 * window_size + negative_sample_size

        self.model = SkipGramModel(vocab_size = self.vocab_size + 1, hidden_size = hidden_size)

    @property
    def features(self):
        return self.model.U.weight.detach()

    def get_similar_word(self, query, k=10):
        W = self.features
        x = W[self.vocab.entity2index[query]].unsqueeze(0)

        sims = cosine_similarity(x, W).squeeze(0)
        sims_idx = sims.argsort().tolist()

        topk = []
        for i in range(1, k + 1):
            idx = sims_idx[-i - 1]
            topk.append((self.vocab.index2entity[idx], sims[idx]))

        return topk

    def fit(self, iterations = 10, batch_size = 512, num_workers = 8, shuffle = True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = SkipGramDataset(center = self.vocab.center,
                                  context = self.vocab.context,
                                  negative_samples = self.vocab.negative_samples,
                                  max_len = self.max_len,
                                  padding_idx = self.padding_idx)
        data_loader = DataLoader(dataset = dataset,
                                 batch_size = batch_size,
                                 num_workers = num_workers,
                                 shuffle = shuffle)
        print(f'device : {device}')

        self.model.to(device)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr = 0.05)

        for ep in range(iterations):
            for i, (center_ids, context_neg_ids, label, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                dot_product = self.model.forward(center_ids.to(device), context_neg_ids.to(device))

                loss = nn.BCEWithLogitsLoss(mask.to(device))
                l = loss(dot_product, label.to(device))

                l.backward()
                optimizer.step()

                if i % 200 == 0:
                    print(l)
            print(f"{ep}-th epoch ...")

        self.model.eval()

    def batchify(self):
        """deprecated. (replaced by SkipGramDataset)"""

        self.batch = []

        for start in range(0, len(self.vocab.center), self.batch_size):
            center_batch = self.vocab.center[start:start + self.batch_size]
            context_neg_batch = self.vocab.context[start:start + self.batch_size]
            label_batch = []
            mask_batch = []

            max_len = max(len(c) for c in context_neg_batch) + self.negative_sample_size

            for i in range(len(center_batch)):
                context_length = len(context_neg_batch[i])
                context_neg_batch[i].extend(self.negative_samples[start + i])

                padding_length = max_len - context_length - self.negative_sample_size
                context_neg_batch[i] += [self.padding_idx] * padding_length

                label = [1.] * (context_length) + [0.] * self.negative_sample_size + [0.] * padding_length
                mask = [1.] * (context_length + self.negative_sample_size) + [0.] * padding_length

                label_batch.append(label)
                mask_batch.append(mask)

            self.batch.append([torch.tensor(center_batch), torch.tensor(context_neg_batch)
                                  , torch.tensor(label_batch), torch.tensor(mask_batch)])

        print('> Batch (center)      :', self.batch[0][0].shape)
        print('> Batch (context_neg) :', self.batch[0][1].shape)
        print('> Batch (label)       :', self.batch[0][2].shape)
        print('> Batch (mask)        :', self.batch[0][3].shape)


if __name__ == '__main__':
    from node2vec import Node2Vec
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

    print(model.get_similar_word('5'))