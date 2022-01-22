# %%
from doctest import OutputChecker
from lib2to3.pgen2 import token
import pandas as pd
import re
from ast import literal_eval

import os
import dill
from tqdm import tqdm
def write_file(output_path, obj):
    ## Write to file
    if output_path is not None:
        folder_path = os.path.dirname(output_path)  # create an output folder
        if not os.path.exists(folder_path):  # mkdir the folder to store output files
            os.makedirs(folder_path)
        with open(output_path, 'wb') as f:
            dill.dump(obj, f)
    return True
def read_file(path):
    with open(path, 'rb') as f:
        generator = dill.load(f)
    return generator

df = pd.read_csv("songs_with_lyrics.csv").drop(columns=["Unnamed: 0"])
df['Lyrics'] = df['Lyrics'].map(literal_eval)

def count_words(s):
    ls = re.split('; |, | |\n', s)
    return len(re.split('; |, | |\n', s))

# Remove entries which are not songs or are very long
ids = [609, 617, 1760, 2014, 2097, 2186, 2247, 2253, 2265, 3157, 4190, 5453, 5725, 5943,
        5966, 6453, 7105, 8053, 8192, 8249, 8345, 8759, 9287, 9384, 9388, 9679, 10006, 10052,
        10202, 10437, 10441, 10448, 10458, 10465, 10475, 10506, 10507, 11083, 11705, 12256]
df = df[~df.Id.isin(ids)]
words = df['Lyrics'].map(count_words)
df = df[words >= 50]
df = df.reset_index(drop=True)

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def LSA(df):
    tfidf = TfidfVectorizer(min_df=4, max_df=0.8, stop_words="english")
    token_data = tfidf.fit_transform(df['Lyrics'])

    svd = TruncatedSVD(n_components=500)
    token_features = svd.fit_transform(token_data)
    
    return token_features

# %%
import string
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stop_words.remove('but')
stop_words.remove('why')
stop_words.remove('don\'t')
stop_words.remove('this')
stop_words.add('it')

# Add one, would, 
punctuation = set(string.punctuation).union({'``', '...', '\'\'', '....', '.....', '--', '’', '–', '—'})
contractions = {'n\'t', '\'s', '\'ve', '\'ll', '\'re', '\'d', '\'m'}
negators = {'not', 'no', 'nothing'}

def tokenize(df):
    tokenized_texts = []
    for i,row in df.iterrows():
        tokens = [token.replace('—', '').replace('\'', '') for token in word_tokenize(row["Lyrics"].lower()) if not (token in stop_words or token in punctuation or token in contractions or token in negators)]
        tokenized_texts.append(tokens)
    return tokenized_texts

def Doc2Features(df):
    tokenized_texts = tokenize(df)
    tagged_lyrics = [TaggedDocument(d[:-2], [i]) for i, d in enumerate(tokenized_texts)]
    model = Doc2Vec(tagged_lyrics, vector_size=1000, window=2, min_count=1, workers=4)
    vectors = np.vstack(tuple(model.dv[i] for i in model.dv.index_to_key))
    return vectors

# %%
import torch
from transformers import BertTokenizer, BertModel

def text_to_embedding(tokenizer, model, in_text):
    '''
    Uses the provided BERT 'model' and 'tokenizer' to generate a vector
    representation of the input string, 'in_text'.

    Returns the vector stored as a numpy ndarray.
    '''

    # ===========================
    #   STEP 1: Tokenization
    # ===========================

    MAX_LEN = 510
    device = 'cuda:7'
    device = torch.device(device)
    print(device)

    # tokenizer will:
    #  (1) Tokenize the sentence
    #  (2) Prepend the '[CLS]' token to the start.
    #  (3) Append the '[SEP]' token to the end.
    #  (4) Map tokens to their IDs.
    print("Tokenizing...")
    output_path = "./output/bert_tokens"
    if os.path.isfile(output_path):
        results = read_file(output_path)
    else:
        results = tokenizer(
            in_text,                         # sentence to encode.
            add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
            truncation=True,                 # Truncate all sentences.
            max_length = MAX_LEN,            # Length to truncate to.
            padding=True,                    # Pad to the longest sequence
            return_attention_mask=True,      
        )
        write_file(output_path, results)

    input_ids = results.input_ids
    attn_mask = results.attention_mask

    # Cast to tensors.
    input_ids = torch.tensor(input_ids)
    attn_mask = torch.tensor(attn_mask)

    # Add an extra dimension for the "batch" (even though there is only one
    # input in this batch)
    # input_ids = input_ids.unsqueeze(0)
    # attn_mask = attn_mask.unsqueeze(0)


    # ===========================
    #   STEP 1: Tokenization
    # ===========================

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Copy the inputs to the GPU
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    model = model.to(device)

    # telling the model not to build the backward graph will make this
    # a little quicker.
    with torch.no_grad():

        # Forward pass, returns hidden states and predictions
        # This will return the logits rather than the loss because we have
        # not provided labels.
        print("Running model...")
        print(input_ids.size(), attn_mask.size())

        split_size = input_ids.size()[0]//64
        full_input_ids = torch.split(input_ids, split_size)
        full_attn_mask = torch.split(attn_mask, split_size)

        token_vecs_batches = []
        for batch in range(len(full_input_ids)):
            input_id_batch = full_input_ids[batch]
            attn_mask_batch = full_attn_mask[batch]

            outputs = model(
                input_ids = input_id_batch,
                token_type_ids = None,
                attention_mask = attn_mask_batch)
            
            print("Processing batch {}".format(batch))
            hidden_states = outputs[2]

            # Sentence Vectors
            # To get a single vector for our entire sentence we have multiple 
            # application-dependent strategies, but a simple approach is to 
            # average the second to last hiden layer of each token producing 
            # a single 768 length vector.
            # `hidden_states` has shape [13 x batch_size x 510 x 768]

            # Take mean of each vector for each token in the sequence after concatenating last 4 hidden layers
            # `token_vecs` is a tensor with shape [batch_size x (768*4)]
            token_vecs_batch = torch.mean(torch.cat((hidden_states[-4], hidden_states[-3], hidden_states[-2], hidden_states[-1]), dim=2), dim=1)
            # print("Token vecs size:", token_vecs_batch.size())

            token_vecs_batches.append(token_vecs_batch)

        # Stack all batches
        sentence_embedding = torch.cat(token_vecs_batches)
        print("Embedding size:", sentence_embedding.size())
            
        # Move to the CPU and convert to numpy ndarray.
        sentence_embedding = sentence_embedding.detach().cpu().numpy()

        return sentence_embedding

def Bert2Vec(df):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    # model.cuda()

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    vectors = text_to_embedding(tokenizer, model, df['Lyrics'].tolist())
    return vectors

# %%
output_path = "./output/bert2vec"
if os.path.isfile(output_path):
    token_features = read_file(output_path)
else:
    # token_features = LSA(df)
    # token_features = Doc2Features(df)
    token_features = Bert2Vec(df)
    write_file(output_path, token_features)


# %%
from sklearn.neighbors import NearestNeighbors

id_to_title = {idx: row["Title"] for idx, row in df.iterrows()}
neigh = NearestNeighbors(n_neighbors=6).fit(token_features)
distances, neighbors = neigh.kneighbors(token_features)
print(distances.shape, neighbors.shape)

# %%
i = np.random.choice(range(neighbors.shape[0]))
for j,neighbor in enumerate(neighbors[i]):
    print(df.iloc[neighbor][["Title", "Artist", "Lyrics"]], distances[i][j])
    # titles = list(map(id_to_title.get, neighbors[i]))
    # print("{}: {}, {}, {}".format(i, titles[0], titles[1:4], distances[i][1:4]))
    print()

# %%
i = df[df["Id"] == 2236].index[0]
print(list(df.index)[-1], len(list(df.index)))
for j,neighbor in enumerate(neighbors[i]):
    print(df.iloc[neighbor][["Title", "Artist", "Lyrics"]], distances[0][j])
    # titles = list(map(id_to_title.get, neighbors[i]))
    # print("{}: {}, {}, {}".format(i, titles[0], titles[1:4], distances[i][1:4]))
    print()

# %%
nearest_neighbor_distances = []
token_features = torch.tensor(token_features).to(torch.device("cuda:1"))
for i in tqdm(range(len(token_features))):
    distances = []
    u = token_features[i].view(1, 1, -1)
    for j in range(len(token_features)):
        if i != j:
            v = token_features[j].view(1, 1, -1)
            distances.append(torch.cdist(u, v).flatten().item())
    df = pd.DataFrame(data={"distance": distances})
    nearest_neighbor_distances.append(list(df['distance'].nsmallest(10)))

df = pd.DataFrame(data=nearest_neighbor_distances, columns=range(1,11))
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print(df.describe())

""" Vector Distances (all pairs)
Mean: 16.166
Min: 0.000
1st: 6.836
5th: 8.449
10th: 9.568
25th: 11.686
50th: 14.961
75th: 20.080
90th: 24.584
95th: 27.120
99th: 31.181
Max: 46.106
"""
""" 10 nearest neighbors of each song
              1          2          3          4          5          6          7          8          9          10
mean      7.1129     7.5079     7.6388     7.7229     7.7873     7.8381     7.8816     7.9201     7.9548     7.9877
std       1.9810     1.7959     1.8024     1.8140     1.8252     1.8365     1.8461     1.8543     1.8635     1.8749
min       0.0000     0.9644     2.5046     2.9372     3.6165     3.8429     3.8660     3.9146     4.0656     4.1007
25%       5.7710     6.2108     6.3353     6.4016     6.4565     6.4979     6.5367     6.5722     6.6004     6.6245
50%       7.2258     7.4985     7.6294     7.7009     7.7660     7.8182     7.8560     7.8954     7.9274     7.9588
75%       8.3227     8.5920     8.7131     8.8029     8.8685     8.9243     8.9712     9.0175     9.0473     9.0784
max      22.7241    22.8830    22.9643    22.9699    22.9905    23.2893    23.2904    23.3039    23.3858    23.3885
"""
# 


