from os.path import join, dirname
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from app.main.utils import logger
from app.main.ai import helper

from app.main import constants


def train_intent_model():
    logger.info('Start intent model training --------->>>>>>>>>')
    max_length = 10
    glove_dimension = 100
    epochs = 300

    try:
        glove_path = constants.GLOVE_PATH
        verbose = constants.VERBOSE
        intents_path = join(dirname(__file__), 'data', constants.INTENTS_PATH)

        glove_dict = helper.generate_glove_dict(glove_path)
        training_data, classes = helper.get_training_data_from_json(intents_path)

        # generate labels from training data
        training_data = helper.convert_y_data_to_labels(training_data, classes)

        # convert to numpy array
        training_data = np.array(training_data)
        labels = training_data[:, 1]
        utterances = training_data[:, 0]

        # prepare tokenizer
        tk = Tokenizer()
        tk.fit_on_texts(utterances)
        vocab_size = len(tk.word_index) + 1

        # integer encode utterances
        encoded_utterances = tk.texts_to_sequences(utterances)
        print(encoded_utterances)
        padded_utterances = pad_sequences(encoded_utterances, maxlen=max_length, padding='post')
        print(padded_utterances)

        embed_matrix = helper.get_embedding_matrix(glove_dict, tk.word_index.items(), vocab_size, glove_dimension)

        model = helper.get_glove_model(vocab_size, glove_dimension, embed_matrix, max_length, len(classes))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        # train the model
        print('x_train: ', padded_utterances)
        print('labels:', labels)
        model.fit(padded_utterances, labels, epochs=epochs, verbose=verbose)

        # save model weights
        helper.save_model(model)

        # save tokenizer data
        helper.save_tokenizer_data(tk.word_index, classes)

        print('================>>>>>>>>>>>>>>>>TRAINING DONE<<<<<<<<<<<<<<<<<=============')

    except Exception as err:
        raise err


def train_dialog_model():
    logger.info('Start train dialog model ----------->>>>>>>>')

    try:
        # TODO: make dialog trainer
        pass
    except Exception as err:
        raise err
