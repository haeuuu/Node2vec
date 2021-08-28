from itertools import chain
from collections import Counter

import torch
import numpy as np

class Vocabulary:
    def __init__(self,
                 data,
                 max_vocab_size = -1,
                 min_counts = 0,
                 negative_sample_size = 5,
                 window_size = 5,
                 sampling = False,
                 subsampling_threshold = 1e-5):
        """
        Parameters
        ----------
        data : list
            [[token6, token1, ...], [token5, token2, ...] ... ]
        max_vocab_size : int
            vocab의 최대 크기. -1인 경우 min_counts만 고려
            양수인 경우 빈도 내림차순으로 정렬한 후 상위 max_vocab_size 개만 선택 
        min_counts : int 
            min_counts 미만으로 등장한 단어는 학습에서 제외
        """

        self.raw_data = data
        self.max_vocab_size = max_vocab_size
        self.min_counts = min_counts
        self.window_size = window_size
        self.sampling = sampling
        self.subsampling_threshold = subsampling_threshold
        self.negative_sample_size = negative_sample_size

        print("Build Vocabulary ...")
        self.build_vocab()

        print("Subsampling ...")
        self.subsampling()

        print("Build Context set ...")
        self.build_context_set()

        print("Negative Sampling ...")
        self.neg_candidates = []
        self.negative_sampling()

    def build_vocab(self):
        most_common = Counter(chain.from_iterable(self.raw_data)).most_common()
        if self.max_vocab_size > 0:
            most_common = most_common[:self.max_vocab_size]
        self.freq = {word: count for word, count in most_common if count >= self.min_counts}

        filtered_data = []
        for line in self.raw_data:
            tmp = []
            for word in line:
                if self.freq.get(word) is None:
                    continue
                tmp.append(word)
            filtered_data.append(tmp)
        self.raw_data = filtered_data

        self.freq = Counter(self.freq)
        vocab = [i for i, j in self.freq.most_common()]

        self.index2entity = dict(zip(range(len(vocab)), vocab))
        self.entity2index = dict(zip(vocab, range(len(vocab))))
        self.vocab_size = len(vocab)

        print("> Vocab size :", self.vocab_size)
        print('> Most common words :', self.freq.most_common(10))

    def _get_drop_prob(self):
        total = sum(self.freq.values())
        drop_prob = lambda x: max(0, 1 - np.sqrt(self.subsampling_threshold / x))
        self.drop_rate = {word: drop_prob(freq / total) for word, freq in self.freq.items()}

    def _drop(self, word):
        return np.random.uniform(0, 1) < self.drop_rate[word]

    def subsampling(self):
        if not self.sampling:
            self.data = self.raw_data
            return

        self._get_drop_prob()

        subsampling = []
        for line in self.raw_data:
            tmp = []
            for word in line:
                if not self._drop(word):
                    tmp.append(word)
            subsampling.append(tmp)
        self.data = subsampling

    def build_context_set(self):
        self.center = []
        self.context = []

        for line in self.data:
            if len(line) < 2:
                continue
            for i in range(len(line)):  # center가 i일 때
                context = []
                center = line[i]
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j < 0 or j >= len(line) or j == i:
                        continue
                    context.append(self.entity2index[line[j]])
                if context:
                    self.context.append(context)
                    self.center.append([self.entity2index[center]])

        print('> pairs of target and context words :', len(self.context))
        print("> Center examples :", self.center[:5])
        print("> Context examples :", self.context[:5])

    def _cal_negative_sampling_prob(self):
        """calculate negative sampling probs"""
        sampling_weight = lambda x: x ** (0.75)
        self.negative_sampling_prob = []
        for i in range(self.vocab_size):
            word = self.index2entity[i]
            self.negative_sampling_prob.append(sampling_weight(self.freq[word]))
        self.negative_sampling_prob = torch.tensor(self.negative_sampling_prob)

    def _get_negative_sample(self):
        """return negative sample(index)"""
        if len(self.neg_candidates) == 0:
            self.neg_candidates = torch.multinomial(self.negative_sampling_prob, 10000, replacement=True).tolist()
        return self.neg_candidates.pop()

    def negative_sampling(self):
        """generate negative samples"""
        self._cal_negative_sampling_prob()
        self.negative_samples = []

        for i in range(len(self.center)):
            negative_ids = [self.center[i]] + self.context[i]
            remove_soon = len(negative_ids)
            while len(negative_ids) < self.negative_sample_size + remove_soon:
                neg = self._get_negative_sample()
                if neg in negative_ids:
                    continue
                negative_ids.append(neg)
            negative_ids = negative_ids[remove_soon:]
            self.negative_samples.append(negative_ids)

        print('> Negative sample :', self.negative_samples[:5])