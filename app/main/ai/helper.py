from os.path import join, dirname
import numpy as np
import json
import pickle
from tensorflow.python.keras import Sequential, layers

from app.main.utils import logger

NLU_MODEL_PATH = join(dirname(__file__), 'models', 'nlu', 'cpd_wl_1.h5')
DIALOG_MODEL_PATH = join(dirname(__file__), 'models', 'dialog', 'cpd_dm_1.h5')
TOKENIZER_DATA_PATH = join(dirname(__file__), 'data', 'tokenizer_data.pkl')


def save_model(model):
    try:
        model.save(NLU_MODEL_PATH)
    except Exception as err:
        raise err


def get_embedding_matrix(glove_dict, token_items, vocab_size, vector_dim):
    marix = np.zeros((vocab_size, vector_dim))

    for word, idx in token_items:
        vector = glove_dict.get(word)
        if vector is not None:
            marix[idx] = vector

    return marix


def generate_glove_dict(path):
    glove_index = dict()
    gl_path = join(dirname(__file__) + '/data/', path)

    file = open(gl_path)

    for line in file:
        splited_line = line.split()
        word = splited_line[0]
        coeficient = np.array(splited_line[1:], dtype='float32')

        glove_index[word] = coeficient

    file.close()
    logger.info('Glove file loaded. Glove dict generated')

    return glove_index


def get_training_data_from_json(path):
    train_data = []
    classes = []

    training_data = json.loads(open(path).read())

    for batch in training_data['intents']:
        for pattern in batch['patterns']:
            train_data.append([pattern, batch['tag']])

        # generate  unique classes array
        if batch['tag'] not in classes:
            classes.append(batch['tag'])

    logger.info('Got training data from Json -------------->>>>')
    return train_data, classes


def convert_y_data_to_labels(data, classes):
    for set in data:
        set[1] = classes.index(set[1])
    return data


def get_glove_model(vocab_size, glove_dimension, embed_matrix, max_length):
    model = Sequential()
    embed = layers.Embedding(vocab_size, glove_dimension, weights=[embed_matrix], input_length=max_length,
                             trainable=False)

    model.add(embed)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def save_tokenizer_data(word_index, classes):
    try:
        pickle.dump({'word_index': word_index, 'classes': classes}, open(TOKENIZER_DATA_PATH, 'wb'))
        logger.info('Pickle saved tokenizer data')
    except Exception as err:
        raise err
