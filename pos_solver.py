###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
"""
Final accuracies: (there might be slight variation in Complex due to the random nature of sampling, it was between 52-53% always)

==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.92%               47.45%
            2. HMM:       94.96%               54.25%
        3. Complex:       94.46%               52.85%

A brief description of the variables used:

1)initial_tag : This stores the P(S) such that S occurs at the first place in the sentence.
2)transition_prob : This stores the P(S_i+1 | S_i) which is the probability of transitioning from tag S_i to S_i+1 when they are consecutive. The keys of this dictionary are the tuples (S_i+1, S_i)
3)emission_prob : This stores the P(W_j | S_i) which is the probability of the word W_j being observed given the tag is S_i. They keys of this dictionary are the tuples (W_j, S_i)
4)tag_count : This stores the count of tag (S) in the training dataset.
5)long_trans : This stores the P(S_i+2 | S_i) which is the probability of transitioning from tag S_i to S_i+2 given there is any other tag in between. The keys of this dictionary are the tuples (S_j, S_i)
6)trans_complex : This is the same as transition_prob. A seperate one was made just to be sure.
7)word_count : This stores the total number of words.

An assumptions made:
--If (W,S), (S_i+1,S_i) or (S_i+2,S_i) doesn't appear in the dictionaries a small probability of 10^-10 was assigned. Laplace smoothing was tried, however the results weren't impressive.
--To calculate the log of the posterior probability we have taken the base as 10.

A brief description of all the models and what their functions do:

1)Simple model:
--This model uses the structure that each observed variable depends only on the hidden variable and nothing else. Also, there is no correlation between the hidden states.
--Thus to get the POS sequence, for each test sentence, we calculate the emission probabilities given all the other POS. Then the tag corresponding to the maximum probability is assigned as the tag for that word.
--The posterior probability is the log ( P(S_1) * P(W_1 | S_1) * P(S_2) * P(W_2 | S_2) ... ). Which is the sum of the log of the emission of W_i | S_i multiplied by the prior of S_i for all i.

2)Viterbi model:
--The Viterbi algorithm was implemented using the paradigm of dynamic programming for better efficiency. A list v[] was maintained for the same.
--For the first word, initial_tag dictionary is used and the v[0] is updated as follows:
    v[0]={S_i:[P(S_i)*P(W_1 | S_i), S_i]}
    where S_i are all the tags possibles
--For all the remaining words v[i] is updated as follows:
    v[i]={S_i: [arg max j (v[i-1]*P(S_j+1 | S_j) * P(W_i | S_i), S_j]}
    where S_j refers to the tag of the W_i-1  (previous word) which has the highest probability for the present word.
--After getting all the different probabilities, we backtrack through v[] and since we have the stored S_j label, we can get the best possible POS tags for the sentence.

3)Complex Model:
--The MCMC sampling was implemented for this model as we couldn't apply Viterbi algorithm here. Thus we need to sample from the posterior
--We first start with an assignment of all nouns and pass it to the generate_sample function.
--This function iterates over all the words in the sentence and assigns each word all the possible tags.
--The probability of that tag occuring is calculated by using the model. For the word W_i it is:
        P(S_i | S_i-1, S_i+1, W_i) = P(S_i | S_i-1) * P(S_i+1 | S_i) * P(W | S_i) * P(S_i)
--The tag which has the maximum probability is set in a random manner using random numbers.
--This is done for all the words in the sentence. We have set a burn in size of 100 and generate 400 samples.
--The tag which appears the most at a particular position is assigned to the final sample, which is then returned as the output.
--The posterior probability is the log ( P(S_1) * P(W_1 | S_1) * P(S_2|S_1) * P(W2|S2) * P(S_2) * P(S_3|S_1,S_2) * P(W_3 | S_3) * P(S_3) ... )
"""
####

import random
import math
from collections import defaultdict
import copy
from collections import Counter
from collections import deque
import sys


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    initial_tag = defaultdict(lambda :0)
    transition_prob = defaultdict(lambda :0)
    emission_prob = defaultdict(lambda :0)
    word_count = 0
    tag_count = defaultdict(lambda :0)

    long_trans = defaultdict(lambda: 0)
    trans_complex = defaultdict(lambda: 0)

    def posterior(self, model, sentence, label):
        if model == "Simple":
            post = 0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.emission_prob:
                    post += math.log(self.emission_prob[(sentence[i], label[i])] * self.tag_count[label[i]] / self.word_count, 10)
                else:
                    post += math.log(0.00000000001 * self.tag_count[label[i]] / self.word_count, 10)
            return post

        elif model == "Complex":
            post = 0
            if (sentence[0], label[0]) in self.initial_tag:
                post += math.log(self.initial_tag[(sentence[0], label[0])], 10)
            else:
                post += math.log(0.00000000001, 10)

            if len(sentence) == 1: return post

            if (sentence[1], label[1]) in self.emission_prob:
                    post += math.log(self.emission_prob[(sentence[1], label[1])] * self.tag_count[label[1]] / self.word_count, 10)

            if (sentence[1], sentence[0]) in self.trans_complex:
                post += math.log(self.trans_complex[(sentence[1], sentence[0])],10)
            else:
                post += math.log(0.00000000001, 10)

            if len(sentence) == 2: return post

            if (sentence[2], label[2]) in self.emission_prob:
                    post += math.log(self.emission_prob[(sentence[2], label[2])] * self.tag_count[label[2]] / self.word_count, 10)
            if (sentence[2], sentence[1]) in self.trans_complex:
                post += math.log(self.trans_complex[(sentence[2], sentence[1])],10)
            else:
                post += math.log(0.00000000001, 10)

            for i in range(3, len(sentence)):
                if (sentence[i], label[i]) in self.emission_prob:
                    post += math.log(self.emission_prob[(sentence[i], label[i])] * self.tag_count[label[i]] / self.word_count, 10)
                else:
                    post += math.log(0.00000000001 * self.tag_count[label[i]] / self.word_count, 10)

                if (sentence[i], sentence[i-1]) in self.trans_complex:
                    post += math.log(self.trans_complex[(sentence[i], sentence[i-1])], 10)
                else:
                    post += math.log(0.00000000001 * self.tag_count[label[i]] / self.word_count, 10)

                if (sentence[i], sentence[i-2]) in self.long_trans:
                    post += math.log(self.long_trans[(sentence[i], sentence[i-2])], 10)
                else:
                    post += math.log(0.00000000001 * self.tag_count[label[i]] / self.word_count, 10)
            return post

        elif model == "HMM":
            post = 0
            if (sentence[0], label[0]) in self.initial_tag:
                post += math.log(self.initial_tag[(sentence[0], label[0])] , 10)
            else:
                post += math.log(0.00000000001, 10)
            for i in range(1, len(sentence)):
                if (sentence[i], label[i]) in self.emission_prob:
                    post += math.log(self.emission_prob[(sentence[i], label[i])]* self.tag_count[label[i]] / self.word_count, 10)
                else:
                    post += math.log(0.00000000001 * self.tag_count[label[i]] / self.word_count, 10)
                if (sentence[i], sentence[i-1]) in self.transition_prob:
                    post += math.log(self.transition_prob[(sentence[i], sentence[i-1])], 10)
                else:
                    post += math.log(0.00000000001, 10)
            return post
        else:
            print("Unknown algo!")

    def get_long_trans(self, data):

        for i in range(len(data)):
            for j in range(len(data[i][1])):
                if j<len(data[i][1])-1:
                    self.trans_complex[(data[i][1][j + 1], data[i][1][j])] += 1
                    if j<len(data[i][1])-2:
                        self.long_trans[(data[i][1][j + 2], data[i][1][j])] += 1

        for t1 in self.long_trans:
            self.long_trans[t1] /= self.tag_count[t1[1]]
        for t1 in self.trans_complex:
            self.trans_complex[t1] /= self.tag_count[t1[1]]

    def get_all_prob(self, data):
        vocab = set()
        for i in range(len(data)):
            self.initial_tag[data[i][1][0]] += 1
            for j in range(len(data[i][1])):
                vocab.add(data[i][0][j])
                self.tag_count[data[i][1][j]] += 1
                if j < len(data[i][1])-1:
                    self.transition_prob[(data[i][1][j + 1], data[i][1][j])] += 1
                self.emission_prob[(data[i][0][j], data[i][1][j])] += 1
        self.word_count = len(vocab)
        for tuple in self.emission_prob:
            self.emission_prob[tuple] /= self.tag_count[tuple[1]]
        for tuple in self.transition_prob:
            self.transition_prob[tuple] /= self.tag_count[tuple[1]]
        t_i = len(data)
        for pos in self.initial_tag:
            self.initial_tag[pos] = self.initial_tag[pos]/t_i

    def train(self, data):
        self.get_all_prob(data)
        self.get_long_trans(data)

    def simplified(self, sentence):
        total = sum(self.tag_count.values())
        tags = []
        for w in sentence:
            maxProb = -1 * math.inf
            state = str()
            for s in self.initial_tag:
                ratio = (float(self.tag_count[s]) / total)
                if (w, s) in self.emission_prob:
                    p = self.emission_prob[(w, s)] * ratio
                else:
                    p = 1e-20 * ratio

                if p > maxProb:
                    maxProb = p
                    state = s
            tags.append(state)
        return tags

    def generate_sample(self, sentence, sample):
        sentence_len = len(sentence)
        tags = list(self.tag_count.keys())
        for index in range(sentence_len):
            word = sentence[index]
            probabilities = [0] * len(self.tag_count)

            s_1 = sample[index - 1] if index > 0 else " "
            s_3 = sample[index + 1] if index < sentence_len - 1 else " "

            for j in range(len(self.tag_count)):
                s_2 = tags[j]
                ep = self.emission_prob[(word, s_2)] if (word, s_2) in self.emission_prob else 0.00000000001
                j_k = self.trans_complex[(s_3, s_2)] if (s_3, s_2) in self.trans_complex else 0.00000000001
                i_j = self.trans_complex[(s_2,s_1)] if (s_2,s_1) in self.trans_complex else 0.00000000001
                i_k = self.long_trans[(s_3, s_1)] if (s_3,s_1) in self.long_trans else 0.00000000001

                if index == 0:
                    probabilities[j] = j_k * ep * self.tag_count[s_2]/self.word_count
                elif index == sentence_len - 1:
                    probabilities[j] = i_j * ep * self.tag_count[s_1]/self.word_count
                else:
                    probabilities[j] = i_j * j_k * i_k * ep * self.tag_count[s_2]/self.word_count

            s = sum(probabilities)
            probabilities = [x / s for x in probabilities]
            maxpos = probabilities.index(max(probabilities))
            rand = random.random()
            if rand < probabilities[maxpos]:
                sample[index] = tags[maxpos]

        return sample

    def mcmc(self, sentence, sample_count):
        #sample = self.simplified(sentence)
        #sample = self.hmm_viterbi(sentence)
        sample = ["noun"] * len(sentence)
        samples=list()
        samples.append(sample)
        for p in range(sample_count+100):
            sample = self.generate_sample(sentence, sample)
            samples.append(sample)
        return samples[100:]

    def max_marginal(self, sentence):
        sample_count = 400
        samples = self.mcmc(sentence, sample_count)
        final_sample = []

        for i in range(len(sentence)):
            tag_count = dict.fromkeys(self.tag_count.keys(), 0)
            for sample in samples:
                tag_count[sample[i]] += 1
            final_sample.append(max(tag_count, key=tag_count.get))
        return final_sample

    def hmm_viterbi(self, sentence):
        v = list()
        viterbi = defaultdict(lambda: list())
        for tag in self.initial_tag:
            if (sentence[0], tag) in self.emission_prob:
                viterbi[tag] = [self.initial_tag[tag] * self.emission_prob[(sentence[0], tag)], tag]
            else:
                viterbi[tag] = [1e-20, tag]
        v.append(viterbi)

        for i in range(1, len(sentence)):
            viterbi = defaultdict(lambda: list())
            for j in self.initial_tag:
                max = -9999999999
                max_tag = j
                for k in self.initial_tag:
                    p = 1e-20
                    if (j, k) in self.transition_prob:
                        p = v[i - 1][k][0] * self.transition_prob[(j, k)]
                    if max < p and p != 1e-20:
                        max = p
                        max_tag = k
                if max == -9999999999:
                    max = 1e-20

                if (sentence[i], j) in self.emission_prob:
                    viterbi[j] = [max * self.emission_prob[(sentence[i], j)], max_tag]
                else:
                    viterbi[j] = [(max * 1e-20), max_tag]
            v.append(viterbi)

        result = deque()
        max = -99999999999
        max_tag = str()
        last_tag = str()
        for i in v[-1]:
            p, tag = v[-1][i]
            if max < p:
                max, max_tag = v[-1][i]
                last_tag = i
        if len(sentence) > 1:
            result.appendleft(last_tag)
        result.appendleft(max_tag)

        for i in range(len(v) - 2, 0, -1):
            result.appendleft(v[i][max_tag][1])
            max_tag = v[i][max_tag][1]
        return result

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.max_marginal(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")