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

        print('================>>>>>>>>>>>>>>>>NLU TRAINING DONE<<<<<<<<<<<<<<<<<=============')

    except Exception as err:
        raise err


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def train_dialog_model():
    logger.info('Start train dialog model ----------->>>>>>>>')
    domain_tokens = dict()
    training_data = []
    max_length = 1
    num_features = 1
    time_steps = 2

    try:
        #
        # prepare training set with dialog sequences ----------------------------------------------->>>>>>>>>>>
        # #

        # get domain data to make dialog tokenized
        domain_data = helper.get_domain_data()

        for idx, action in enumerate(domain_data['actions_list']):
            # create dict where with action name prop and index as a value (start with 1)
            domain_tokens[action] = (idx + 1)
        dialog_data = helper.get_dialog_flow_data()

        for flow in dialog_data['dialogs']:
            sequence = flow['flow']
            sequence = list(map(lambda sq: domain_tokens[sq], sequence))

            # get max list length
            if len(sequence) > max_length:
                max_length = len(sequence)

            training_data.append(sequence)

        # pad sequences
        padded_flows = pad_sequences(training_data, maxlen=max_length, padding='post')
        # padded_flows = np.divide(padded_flows, divider)

        # prepare training set
        # x_set, y_set = padded_flows[:, :-1], padded_flows[:, 1:]


        # get LSTM model --------------------------------------------------------------------------->>>>>>>>>>>>
        # #
        model = helper.create_dialog_netowrk(time_steps, num_features)
        model.compile(loss='mse', optimizer='adam')

        model.summary()

        # fit model
        for idx, sample in enumerate(padded_flows):
            if idx == 1:
                print(sample)
                x_train, y_train = split_sequence(sample, time_steps)
                print(x_train)
                print(y_train)

                x_0 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                print('reshaped------------->>>>>>>>>')
                print(x_0)

                print('-----------------------------------------')
                model.fit(x_0, y_train, epochs=100, batch_size=1, verbose=0)

        # save model

        # save dialog tokens
        print('================>>>>>>>>>>>>>>>>DIALOG TRAINING DONE<<<<<<<<<<<<<<<<<=============')
        x_test = np.array([10, 21])
        x_test = np.reshape(x_test, (1, time_steps, num_features))

        pred = model.predict(x_test, verbose=0)
        print(pred)



    except Exception as err:
        raise err
