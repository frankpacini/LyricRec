# %%
from doctest import OutputChecker
import pandas as pd
import re
from ast import literal_eval
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer will:
    #  (1) Tokenize the sentence
    #  (2) Prepend the '[CLS]' token to the start.
    #  (3) Append the '[SEP]' token to the end.
    #  (4) Map tokens to their IDs.
    print("Tokenizing...")
    results = tokenizer(
        in_text,                         # sentence to encode.
        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
        truncation=True,                 # Truncate all sentences.
        max_length = MAX_LEN,            # Length to truncate to.
        padding=True,                    # Pad to the longest sequence
        return_attention_mask=True,      
    )
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

        split_size = input_ids.size()[0]//128
        full_input_ids = torch.split(input_ids, split_size)
        full_attn_mask = torch.split(attn_mask, split_size)

        token_vecs_batches = []
        for batch in range(len(full_input_ids)):
            input_id_batch = full_input_ids[batch]
            print(input_id_batch.device)
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
            # `hidden_states` has shape [13 x 1 x ? x 768]

            # `token_vecs` is a tensor with shape [? x 768]
            token_vecs_batch = torch.vstack((hidden_states[-4][0], hidden_states[-3][0], hidden_states[-2][0], hidden_states[-1][0]))
            print(token_vecs_batch.size())

            token_vecs_batches.append(token_vecs_batch)

        # Calculate the average of all ? token vectors.
        sentence_embedding = torch.mean(torch.stack(token_vecs_batches), dim=0)
            
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
import os
import dill
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

# %%
i = np.random.choice(range(neighbors.shape[0]))
for i,neighbor in enumerate(neighbors[i]):
    print(df.iloc[neighbor][["Title", "Artist", "Lyrics"]], distances[neighbor][i])
    # titles = list(map(id_to_title.get, neighbors[i]))
    # print("{}: {}, {}, {}".format(i, titles[0], titles[1:4], distances[i][1:4]))
    print()

# %%
for i,neighbor in enumerate(neighbors[0]):
    print(df.iloc[neighbor][["Title", "Artist", "Lyrics"]], distances[neighbor][i])
    # titles = list(map(id_to_title.get, neighbors[i]))
    # print("{}: {}, {}, {}".format(i, titles[0], titles[1:4], distances[i][1:4]))
    print()

# %%



