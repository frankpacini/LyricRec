# Code from models.py to load external embeddings

import codecs
import torch
from torch import nn
import numpy as np
import logging

def load_external_embeddings(params, emb_path):
    """
    Reload pretrained embeddings from a text file.
    """
    
    word2id = {}
    vectors = []

    # load pretrained embeddings
    _emb_dim_file = params.emb_dim
    with codecs.open(emb_path) as f:
        for i, line in enumerate(f):
            if len(line.split()) == 2:
                i -= 1
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                vect[0] = 0.01
            assert word not in word2id
            assert vect.shape == (_emb_dim_file,), i
            word2id[word] = len(word2id)
            vectors.append(vect[None])

    logging.info("Loaded %i pre-trained word embeddings" % len(vectors))

    # compute new vocabulary / embeddings
    
    dico = word2id

    # id2word mapping is not necessary and avoids defining a Dictionary class
    # id2word = {v: k for k, v in word2id.items()}
    #dico = id2word.copy()
    #dico = dico.update(word2id)
    
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if params.cuda and torch.cuda.is_available() else embeddings
    assert embeddings.size() == (len(word2id), params.emb_dim), ((len(word2id), params.emb_dim, embeddings.size()))

    return dico, embeddings

########################################################

from collections import Counter
import pandas as pd
import numpy as np
import os
import re
import pickle
import torch
import pprint
pp = pprint.PrettyPrinter(indent=4)

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords
remove_these = set(stopwords.words('english'))

REMOVE_STOPWORDS = True

def get_song_emb_dict(dataset):

    with open('mxm_reverse_mapping.txt','r') as f:        # Load mapping from contracted word to full word string in the mxm dataset
        lines = f.readlines()
        map = {}
        for l in lines:
            input, output = l.split("<SEP>")
            map[input] = output   

    print("Reverse map")

    with open(dataset,'r') as f:
        lines = f.readlines()
        words = lines[17].replace('%','').split(',')      # get list of words which will be referenced by index in the dataset
        songs_dict = {}

        for i,l in list(enumerate(lines))[18:]:     # a line represents data on a song, with the first comma separated value being the MSDID, and the remainder being a word index followed by its count
            song_info = l.split(',')
            MSDID = song_info[0]
            song_bow = [x.split(':') for x in song_info[2:]]
            song_dict = {}
            for word, word_count in song_bow:
                song_dict[int(word)] = int(word_count.replace('\n',''))

            # word_lists = [[words[word-1]]*song_dict[word] for word in song_dict.keys()]

            song = [  (map[words[word-1].replace('\n','')].replace('\n',''),  song_dict[word]) 
                    for word in song_dict.keys()]
            if REMOVE_STOPWORDS:
                # song = [(map[w[0].replace('\n','')].replace('\n',''),w[1]) for w in song if w[0] not in remove_these]
                song = [s for s in song if s[0] not in remove_these]      # Filter out words considered frequent words in the ntlk stopwords dataset
            
            songs_dict[str(MSDID)] = song     # songs_dict: MSDID -> (word, count) list

    print("Training set")

    song_msd_ids = list(songs_dict.keys())
    # print(all_songs_dict[song_msd_ids[2]])

    pkl_file = open("ft_params.pkl", 'rb')
    params = pickle.load(pkl_file)
    # mapping = Generator(params)
    # checkpoint = torch.load("./models/ft_model.t7")
    # mapping.load_state_dict(checkpoint['model'])
    # mapping.eval()
    out_dico, out_emb = load_external_embeddings(params, "ft_postspec.txt")         # out_dico: word -> idx, out_emb: idx -> vector

    print("Fast text embeddings")

    all_song_emb_dict = {} # {id: [(out_emb[out_dico["en_" + tup[0]]], tup[1]) for tup in songs_dict[id] if "en_" + tup[0] in out_dico] for id in song_msd_ids}

    # Generate set of all words using dictionary (mapping to global word frequency)
    word_set = {}                     
    for id in song_msd_ids:
        for tup in songs_dict[id]:
            if tup[0] in word_set:
                word_set[tup[0]] = word_set[tup[0]] + tup[1]
            else:
                word_set[tup[0]] = tup[1]

    # Find the most frequent words not in the AuxGAN set
    """
    with torch.no_grad():
        mapped_emb = mapping(out_emb).data.cpu().numpy()

    missed_words = {}
    for id in song_msd_ids:
        for tup in songs_dict[id]:
            if "en_" + tup[0] not in out_dico:
                if tup[0] not in missed_words:
                    missed_words[tup[0]] = tup[1]
                else:
                    missed_words[tup[0]] = missed_words[tup[0]] + tup[1]
    pp.pprint(sorted(missed_words.items(), key=lambda x: x[1], reverse=True))
    """
  
    filtered_words = [word for word in word_set.keys() if ("en_" + word) in out_dico]             # Filter out words not in the AuxGAN set
    # emb_list = [(out_emb[out_dico["en_" + word]], word_set[word]) for word in filtered_words]
    emb_dict = {word: out_emb[out_dico["en_" + word]] for word in filtered_words}

    return songs_dict, song_msd_ids, emb_dict

# def get_all_word_emb():

songs_dict, ids, word_emb_dict = get_song_emb_dict('mxm_dataset_train.txt') 
print("Song dict")

# Import mxm 779 to get id to name mapping later

# all_song_meta_dict = dict()
# with open('../../data/mxm_779k_matches.txt','r') as f:
#     lines = f.readlines()
#     for i in range(18, len(lines)):
#         line = lines[i].split('<SEP>')
#         MSDID = line[0]
#         artist = line[1]
#         title = line[2]
#         all_song_meta_dict[str(MSDID)] = {'artist': artist, 'title': title}


###############################################################################

import torch
from sklearn.mixture import GaussianMixture

words = list(word_emb_dict.keys())
word2idx = {word: i for i,word in enumerate(words)}

K = 39

tensor_embeddings = torch.cat([torch.unsqueeze(word_emb_dict[word],0) for word in words], 0) # (M, N)

np_embeddings = torch.cat([torch.unsqueeze(word_emb_dict[word],0) for word in words], 0).cpu().numpy()

gm = GaussianMixture(n_components=K, random_state=0).fit(np_embeddings)
classes = torch.tensor(gm.predict(np_embeddings), device="cuda") # (M,)
probs = torch.tensor(gm.predict_proba(np_embeddings), device="cuda") # (M, K)

############################################################################

# Failed attempt to manually implement Gaussian Mixture Model

"""
from torch.distributions import multivariate_normal
from sklearn.datasets import make_spd_matrix
# K = 3
# epsilon = 1e-8
# iters = 40
def gmm(data):
    # X = np.array(data)
    X = data
    M = X.shape[0]
    N = X.shape[1]
    # weights = np.ones((K)) / K
    weights = torch.ones((K)) / K
    # means = np.random.choice(X.flatten(), (K,X.shape[1]))
    perm = torch.randperm(X.numel())
    idx = perm[:K*N]
    means = torch.reshape(X.flatten()[idx], (K,N))
    # cov = np.array([make_spd_matrix(X.shape[1]) for _ in range(K)])
    cov = torch.tensor([make_spd_matrix(N) for _ in range(K)], device='cuda').float()
    scale_tril = torch.tril(cov)

    bayes = []

    for step in range(iters):
        # likelihood = torch.tensor([multivariate_normal.pdf(x=X, mean=means[j], cov=cov[j]) for j in range(K)])
        print(multivariate_normal.MultivariateNormal(means[2], covariance_matrix=cov[2]).log_prob(X))
        likelihood = torch.cat([torch.unsqueeze(multivariate_normal.MultivariateNormal(means[j], covariance_matrix=cov[j]).log_prob(X), 0) for j in range(K)], 0)
        assert likelihood.shape == (K, M)

        bayes = []
        for j in range(K):
            bayes.append((likelihood[j] * weights[j]) / (torch.sum(torch.cat([torch.unsqueeze(likelihood[i] * weights[i], 0) for i in range(K)], 0), axis=0)+epsilon))

            means[j] = torch.sum(torch.reshape(bayes[j], (M, 1)) * X) / (torch.sum(bayes[j]+epsilon))
            cov[j] = torch.mm((torch.reshape(bayes[j], (M, 1)) * (X - means[j])).T, (X - means[j])) / (torch.sum(bayes[j])+epsilon)
            
            weights[j] = torch.mean(bayes[j])

            assert cov.shape == (K, N, N)
            assert means.shape == (K, N)

        
    
    classes = []
    for i in range(M):
        # likelihood = torch.tensor([multivariate_normal.pdf(x=X[i,:], mean=means[j], cov=cov[j]) for j in range(K)])
        likelihood = torch.tensor([multivariate_normal.MultivariateNormal(means[j], cov[j]).log_prob(X[i,:]) for j in range(K)])
        print(i, likelihood)
        classes.append(torch.argmax(likelihood)+1)
    
    return classes, weights, means, cov
"""

############################################################################

# Determine the optimal number of clusters, evaulated with Akaike information criterion on test set
_, _, word_emb_dict_test = get_song_emb_dict('mxm_dataset_test.txt')

np_embeddings_test = torch.cat([torch.unsqueeze(word_emb_dict_test[word],0) for word in words], 0).cpu().numpy()


aic = []
clusters = range(31,35)

for K in clusters:
  gm = GaussianMixture(n_components=K, random_state=0).fit(np_embeddings)
  aic.append(gm.aic(np_embeddings_test))
  print(K)

print(aic)

"""
AIC for train
clusters = [10, 30, 50, 80, 100, 150, 200]
aic = [-4504852.787373134, -6666515.6722224485, -6703928.966121189, -5264487.663826261, -3913524.8514031116, 126418.23396109417, 4308583.780426782]

clusters = [10, 20, 30, 40, 50, 60, 70, 80]
aic = [-4504852.787373134, -5597897.493379482, -6666515.6722224485, -6688446.631000172, -6703928.966121189, -6177245.378351711, -5778542.452237887, -5264487.663826261]

clusters = [30, 35, 40, 45, 50, 55]
aic = [-6666515.6722224485, -6654718.370988596, -6688446.631000172, -6654966.182520773, -6703928.966121189, -6312888.754963251]

clusters = range(46, 55)
aic = [-6613663.237831932, -6690252.560396293, -6707868.594154408, -6695732.93230935, -6703928.966121189, -6605595.489092547, -6576743.819046922, -6733423.197465405, -6717831.644150782]

clusters = range(41,46)
aic = [-6735285.285983134, -6691415.717371201, -6650176.874654621, -6664396.813439354, -6654966.182520773]

"""

"""
AIC for test
clusters = [10, 30, 50, 80, 100, 150, 200]
aic = [-4483687.5748528335, -6462898.944675984, -6580692.25876198, -5107816.440313572, -3772441.9127039015, 169570.8663827367, 4325081.014762333]

clusters = [10, 20, 30, 40, 50, 60, 70, 80]
aic = [-4483687.5748528335, -5597856.083688019, -6462898.944675984, -6746603.306217089, -6580692.25876198, -6266922.301159833, -5729930.116943039, -5107816.440313572]

clusters = [30, 35, 40, 45, 50, 55]
aic = [-6462898.944675984, -6750139.253499374, -6746603.306217089, -6659716.89766887, -6580692.25876198, -6504312.897404702]

clusters = range(36, 45)
aic = [-6668556.477066301, -6829178.28346259, -6829365.392411279, -6810971.983992221, -6746603.306217089, -6795956.870910021, -6743121.8986961655, -6700375.869984798, -6670593.19445217]

clusters = range(31,35)
aic = [-6471749.693533244, -6457655.665048353, -6503741.977487061, -6535845.925271023]

"""

################################################################################

# Obtain class matrices from tensors

id = ids[25]
S = len(songs_dict[id])
count = []
indices = []
for tup in songs_dict[id]:
  indices.append(word2idx[tup[0]])
  count.append(tup[1])

count = torch.tensor(count, device="cuda")
indices = torch.tensor(indices, device="cuda")
lyrics_to_cluster_probs = torch.index_select(probs, 0, indices)
lyrics_to_embs = torch.index_select(tensor_embeddings, 0, indices)
class_matrix = torch.mm(torch.transpose(lyrics_to_cluster_probs, 0, 1).double(), (lyrics_to_embs * count.view(-1, 1).double()))

# S: # of words in song bag-of-words
# count: S x 1 # no of occurences for each word
# lyrics_to_cluster_probs: S x K  Produce by filtering the rows of probs down to only the words in the song
# lyrics_to_embs: S x N  Produce by filtering rows of tensor_embeddings down to only words in the song
# Then the desired K x N class matrix should be = lyrics_to_cluster_probs.T @ (lyrics_to_embs * count)

###########################################################################

# print(len(tensor_embeddings), tensor_embeddings[0].numel())
# numpy_embeddings = tensor_embeddings.cpu().numpy()
# print(numpy_embeddings[0:10])

# classes, weights, means, cov = gmm(tensor_embeddings)
print("GMM")

cluster_set = [[] for _ in range(K)]
cluster_dict = {}

for i,val in enumerate(classes):
    cluster_set[val - 1].append(words[i])
    cluster_dict[words[i]] = val

pp.pprint(cluster_set)