#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


# Set up IPython to show all outputs from a cell
import warnings
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

warnings.filterwarnings('ignore', category=RuntimeWarning)

RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 1024
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.9
LSTM_CELLS = 64
VERBOSE = 2
SAVE_MODEL = True


# In[4]:


# Read in data
url = 'https://raw.githubusercontent.com/joycechen914/SM_Project/master/got_scripts_breakdown.csv'
data = pd.read_csv(url, sep=';')

# Extract sentences
original_sentence = list(data['Sentence'])
len(original_sentence)

data.head()


# In[5]:


from keras.preprocessing.text import Tokenizer

example = original_sentence[1]
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([example])
s = tokenizer.texts_to_sequences([example])[0]
' '.join(tokenizer.index_word[i] for i in s)


# In[6]:


tokenizer = Tokenizer(filters='"#$%&*+/:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([example])
s = tokenizer.texts_to_sequences([example])[0]
' '.join(tokenizer.index_word[i] for i in s)
tokenizer.word_index.keys()


# In[7]:


import re


def format_sentence(sentence):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    sentence = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', sentence)

    # Remove references to figures
    sentence = re.sub(r'\((\d+)\)', r'', sentence)

    # Remove double spaces
    sentence = re.sub(r'\s\s', ' ', sentence)
    return sentence


f = format_sentence(example)
f


# In[8]:


def remove_spaces(sentence):
    """Remove spaces around punctuation"""
    sentence = re.sub(r'\s+([.,;?])', r'\1', sentence)

    return sentence


remove_spaces(' '.join(tokenizer.index_word[i] for i in s))


# In[9]:


formatted = []

# Iterate through all the original sentences
for a in original_sentence:
    formatted.append(format_sentence(a))

len(formatted)
print(formatted[:20])


# In[10]:


def make_sequences(texts,
                   training_length=50,
                   lower=True,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    """Turn a set of texts into sequences of integers"""

    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit to sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]
    over_idx = [
        i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
    ]

    new_texts = []
    new_sequences = []

    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])

    training_seq = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in new_sequences:

        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length:i + 1]

            # Set the features and label
            training_seq.append(extract[:-1])
            labels.append(extract[-1])

    print(f'There are {len(training_seq)} training sequences.')

    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels


# In[11]:


TRAINING_LEGNTH = 50
filters = '!"#$%&()*+/:<=>@[\\]^_`{|}~\t\n'
# word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(
#     data_lemmatized, TRAINING_LEGNTH, lower=True, filters=filters)

word_idx, idx_word, num_words, word_counts, sentences, sequences, features, labels = make_sequences(
    formatted, TRAINING_LENGTH, lower=True, filters=filters)


# In[12]:


n = 3
features[n][:10]


# In[13]:


def find_answer(index):
    """Find label corresponding to features for index in training data"""

    # Find features and label
    feats = ' '.join(idx_word[i] for i in features[index])
    answer = idx_word[labels[index]]

    print('Features:', feats)
    print('\nLabel: ', answer)
    

find_answer(n)


# In[14]:


original_sentence[0]


# In[15]:


find_answer(100)


# In[16]:


sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]


# In[17]:


from sklearn.utils import shuffle


def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=TRAIN_FRACTION):
    """Create training and validation features and labels."""

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid


# In[18]:


X_train, X_valid, y_train, y_valid = create_train_valid(
    features, labels, num_words)
X_train.shape
y_train.shape


# In[19]:


# Load in unzipped file
glove_vectors = '/Users/joycechen/Desktop/ML/glove.6B/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
glove.shape


# In[20]:


vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

del glove

vectors[100], words[100]


# In[21]:


vectors.shape


# In[22]:


word_lookup = {word: vector for word, vector in zip(words, vectors)}

embedding_matrix = np.zeros((num_words, vectors.shape[1]))

not_found = 0

for i, word in enumerate(word_idx.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
    else:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')


# In[23]:


# Normalize and convert nan to 0
embedding_matrix = embedding_matrix /     np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
embedding_matrix = np.nan_to_num(embedding_matrix)


# In[24]:


def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    """Find closest words to a query word in embeddings"""

    idx = word_idx.get(query, None)
    # Handle case where query is not in vocab
    if idx is None:
        print(f'{query} not found in vocab.')
        return
    else:
        vec = embedding_matrix[idx]
        # Handle case where word doesn't have an embedding
        if np.all(vec == 0):
            print(f'{query} has no pre-trained embedding.')
            return
        else:
            # Calculate distance between vector and all others
            dists = np.dot(embedding_matrix, vec)

            # Sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]

    print(f'Query: {query}\n')
    max_len = max([len(i) for i in closest])
    # Print out the word and cosine distances
    for word, dist in zip(closest, sorted_dists):
        print(f'Word: {word:15} Cosine Similarity: {round(dist, 4)}')


# In[25]:


find_closest('the', embedding_matrix, word_idx, idx_word)


# In[26]:


find_closest('throne', embedding_matrix, word_idx, idx_word, 10)


# In[27]:


find_closest('arya', embedding_matrix, word_idx, idx_word, 10)


# In[28]:


from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam

from keras.utils import plot_model


# In[31]:


def make_word_level_model(num_words,
                          embedding_matrix,
                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          bi_direc=False):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of LSTM cell layers."""

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))
    model.add(Dense(128, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = make_word_level_model(
    num_words,
    embedding_matrix=embedding_matrix,
    lstm_cells=LSTM_CELLS,
    trainable=False,
    lstm_layers=1)
model.summary()


# In[34]:


from IPython.display import Image
model_name = 'pre-trained-rnn'
model_dir = '/Users/joycechen/Desktop/ML/SM_Project/models/'

plot_model(model, to_file=f'{model_dir}{model_name}.png', show_shapes=True)

Image(f'{model_dir}{model_name}.png')


# In[46]:


model_name = 'pre-trained-rnn'
model_dir = '/Users/joycechen/Desktop/ML/SM_Project/models/'

from keras.callbacks import EarlyStopping, ModelCheckpoint

BATCH_SIZE = 2048


def make_callbacks(model_name, save=SAVE_MODEL):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks


callbacks = make_callbacks(model_name)


# In[47]:


history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))


# In[48]:


def load_and_evaluate(model_name, return_model=False):
    """Load in a trained model and evaluate with log loss and accuracy"""

    model = load_model(f'{model_dir}{model_name}.h5')
    r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

    valid_crossentropy = r[0]
    valid_accuracy = r[1]

    print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
    print(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model


# In[49]:


model = load_and_evaluate(model_name, return_model=True)


# In[39]:


print(history.history.keys())
len((history.history['val_accuracy']))


# In[42]:


import matplotlib.pyplot as pyplot
# summarize history for accuracy
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epochs')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()
# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epochs')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()


# In[43]:


np.random.seed(40)

# Number of all words
total_words = sum(word_counts.values())

# Compute frequency of each word in vocab
frequencies = [word_counts[word] / total_words for word in word_idx.keys()]
frequencies.insert(0, 0)


# In[44]:


frequencies[1:10], list(word_idx.keys())[0:9]


# In[50]:


print(
    f'The accuracy is {round(100 * np.mean(np.argmax(y_valid, axis = 1) == 1), 4)}%.'
)


# In[51]:


random_guesses = []

# Make a prediction based on frequencies for each example in validation data
for i in range(len(y_valid)):
    random_guesses.append(
        np.argmax(np.random.multinomial(1, frequencies, size=1)[0]))


# In[52]:


from collections import Counter

# Create a counter from the guesses
c = Counter(random_guesses)

# Iterate through the 10 most common guesses
for i in c.most_common(10):
    word = idx_word[i[0]]
    word_count = word_counts[word]
    print(
        f'Word: {word} \tCount: {word_count} \tPercentage: {round(100 * word_count / total_words, 2)}% \tPredicted: {i[1]}'
    )


# In[53]:


accuracy = np.mean(random_guesses == np.argmax(y_valid, axis=1))
print(f'Random guessing accuracy: {100 * round(accuracy, 4)}%')


# In[54]:


from IPython.display import HTML


def header(text, color='black'):
    raw_html = f'<h1 style="color: {color};"><center>' +         str(text) + '</center></h1>'
    return raw_html


def box(text):
    raw_html = '<div style="border:1px inset black;padding:1em;font-size: 20px;">' +         str(text)+'</div>'
    return raw_html


def addContent(old_html, raw_html):
    old_html += raw_html
    return old_html


# In[55]:


import random


def generate_output(model,
                    sequences,
                    training_length=50,
                    new_words=50,
                    diversity=1,
                    return_output=False,
                    n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - training_length - 10)
    # Ending index for seed
    end_idx = seed_idx + training_length

    gen_list = []

    for n in range(n_gen):
        # Extract the seed sequence
        seed = seq[seed_idx:end_idx]
        original_sequence = [idx_word[i] for i in seed]
        generated = seed[:] + ['#']

        # Find the actual entire sequence
        actual = generated[:] + seq[end_idx:end_idx + new_words]

        # Keep adding new words
        for i in range(new_words):

            # Make a prediction from the seed
            preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
                np.float64)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)

            # Softmax
            preds = exp_preds / sum(exp_preds)

            # Choose the next word
            probas = np.random.multinomial(1, preds, 1)[0]

            next_idx = np.argmax(probas)

            # New seed adds on old word
            seed = seed[1:] + [next_idx]
            generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, '< --- >'))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, '< --- >'))

    a = a[training_length:]

    gen_list = [
        gen[training_length:training_length + len(a)] for gen in gen_list
    ]

    if return_output:
        return original_sequence, gen_list, a

    # HTML formatting
    seed_html = ''
    seed_html = addContent(seed_html, header(
        'Seed Sequence', color='darkblue'))
    seed_html = addContent(seed_html,
                           box(remove_spaces(' '.join(original_sequence))))

    gen_html = ''
    gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
    gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

    a_html = ''
    a_html = addContent(a_html, header('Actual', color='darkgreen'))
    a_html = addContent(a_html, box(remove_spaces(' '.join(a))))

    return seed_html, gen_html, a_html


# In[56]:


seed_html, gen_html, a_html = generate_output(model, sequences,
                                              TRAINING_LENGTH)
HTML(seed_html)
HTML(gen_html)
HTML(a_html)


# In[57]:


seed_html, gen_html, a_html = generate_output(
    model, sequences, TRAINING_LENGTH, diversity=1)
HTML(seed_html)
HTML(gen_html)
HTML(a_html)


# In[58]:


def clear_memory():
    import gc
    gc.enable()
    for i in [
            'model', 'X', 'y', 'word_idx', 'idx_word', 'X_train', 'X_valid,'
            'y_train', 'y_valid', 'embedding_matrix', 'words', 'vectors',
            'labels', 'random_guesses', 'training_seq', 'word_counts', 'data',
            'frequencies'
    ]:
        if i in dir():
            del globals()[i]
    gc.collect()


clear_memory()


# In[59]:


TRAINING_LENGTH = 50

filters = '!"%;[\\]^_`{|}~\t\n'
word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(
    formatted, TRAINING_LENGTH, lower=False, filters=filters)


# In[60]:


embedding_matrix = np.zeros((num_words, len(word_lookup['the'])))

not_found = 0

for i, word in enumerate(word_idx.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
    else:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')
embedding_matrix.shape


# In[61]:


# Split into training and validation
X_train, X_valid, y_train, y_valid = create_train_valid(
    features, labels, num_words)
X_train.shape, y_train.shape


# In[62]:


model = make_word_level_model(
    num_words,
    embedding_matrix,
    lstm_cells=LSTM_CELLS,
    trainable=True,
    lstm_layers=1)
model.summary()


# In[63]:


model_name = 'train-embeddings-rnn'

callbacks = make_callbacks(model_name)


# In[64]:


model.compile(
    optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))


# In[65]:


model = load_and_evaluate(model_name, return_model=True)


# In[66]:


seed_html, gen_html, a_html = generate_output(
    model, sequences, TRAINING_LENGTH, diversity=0.75)
HTML(seed_html)
HTML(gen_html)
HTML(a_html)


# In[68]:


# summarize history for accuracy
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epochs')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()
# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epochs')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()


# In[67]:


seed_html, gen_html, a_html = generate_output(
    model, sequences, TRAINING_LENGTH, diversity=0.75)
HTML(seed_html)
HTML(gen_html)
HTML(a_html)


# In[ ]:




