from tswift import Artist, Song
import random

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

import codecs
import pickle
import logging
import nltk
stopwords = nltk.corpus.stopwords
remove_these = set(stopwords.words('english'))

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
    
    dico = word2id
    
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if params.cuda and torch.cuda.is_available() else embeddings
    assert embeddings.size() == (len(word2id), params.emb_dim), ((len(word2id), params.emb_dim, embeddings.size()))

    return dico, embeddings

pkl_file = open("model/ft_params.pkl", 'rb')
params = pickle.load(pkl_file)
out_dico, out_emb = load_external_embeddings(params, "data/ft_postspec.txt")

punctuation = ['.', ',', '?', '!', '(', ')']

artist = Artist('Kendrick Lamar')

word_set = {} # word -> overall count
words = [] # word_idx -> word_str
word2idx = {} # word_str -> word_idx
word_emb_dict = {} # word_str -> embedding

songs = [] # song_idx -> song
song_words = [] # song_idx -> (word, count)
song2idx = {} # song -> song_idx

for song in artist.songs:
#song = artist.songs[163] 
    try:
        lyrics = song.lyrics.replace('\n', ' ')
    except TswiftError:
        continue
        
    for p in punctuation:
        lyrics = lyrics.replace(p, '')

    lyrics = [word.lower() for word in lyrics.split(' ') if word != ""]
    lyrics = [word.replace("\'", "") for word in lyrics if not word in remove_these]

    song_word_set = {}
    for word in lyrics:
        if word in song_word_set:
            song_word_set[word] += 1
        elif "en_" + word in out_dico:
            song_word_set[word] = 1
        if word in word_set:
            word_set[word] += 1
        elif "en_" + word in out_dico:
            word_set[word] = 1
    
    songs.append(song.title)
    song_words.append([(word, song_word_set[word]) for word in list(song_word_set.keys())])
    song2idx[song.title] = len(songs) - 1

for i, word in enumerate(list(word_set.keys())):
    words.append(word)
    word2idx[word] = i
    word_emb_dict[word] = out_emb[out_dico["en_" + word]]

print("Collected data")

train_indices = random.sample(range(len(words)), len(words) * 0.8)
test_indices = set(range(len(words))) - set(train_indices)

words_train = [words[i] for i in train_indices]
words_test = [words[i] for i in test_indices]

tensor_embeddings_train = torch.cat([torch.unsqueeze(word_emb_dict[word],0) for word in words_train], 0) # M: len(words_train) by N: embedding length 
np_embeddings_train = tensor_embeddings_train.cpu().numpy()

tensor_embeddings_test = torch.cat([torch.unsqueeze(word_emb_dict[word],0) for word in words_test], 0) # M: len(words_train) by N: embedding length 
np_embeddings_test = tensor_embeddings_test.cpu().numpy()




"""with open('artists.txt', 'r') as artists:
    for artist_name in artists:
        artist_name = artist_name.rstrip()
        print(artist_name)
        artist = Artist(artist_name)
        print(len(artist.songs))"""

