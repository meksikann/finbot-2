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


def train_dialog_model():
    logger.info('Start train dialog model ----------->>>>>>>>')
    domain_tokens = dict()
    training_data = []
    max_length = 1
    look_back = 1
    time_step = 1

    try:
        #
        # prepare training set with dialog sequences ----------------------------------------------->>>>>>>>>>>
        # #

        # get domain data to make dialog tokenized
        domain_data = helper.get_domain_data()

        for idx, action in enumerate(domain_data['actions_list']):
            # create dict where with action name prop and index as a value (start with 1)
            domain_tokens[action] = idx + 1

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

        # create dataset
        x_set, y_set = padded_flows[:, :-1], padded_flows[:, 1:]

        print('x:', x_set[1])
        print('y:', y_set[1])

        #
        # get LSTM model --------------------------------------------------------------------------->>>>>>>>>>>>
        # #
        # model = helper.create_dialog_netowrk(6)
        # model.compile(loss='mean_squared_error', optimizer='adam')
        #
        # model.summary()

        # fit model
        # print(x_set.shape[0])
        # print(x_set.shape[1])

        # x_set = np.reshape(x_set, (x_set.shape[0], 1, x_set.shape[1]))
        #
        # for i in range(len(x_set)):
        #
        #     # # print('x_set[i].shape[0]', x_set[i].shape)
        #     # x = np.reshape(x_set[i], (x_set[i].shape[0], 1, 1))
        #     # # print('x shape', x)
        #
        #     model.fit(x_set[i], y_set[i], epochs=100, batch_size=1, verbose=1)

        # test_x = np.array([
        #     [3], [3], [3]
        # ])
        #
        # prediction = model.predict(test_x)
        #
        # print('prediction:', prediction)
        # save model

        # save dialog tokens
        print('================>>>>>>>>>>>>>>>>DIALOG TRAINING DONE<<<<<<<<<<<<<<<<<=============')

    except Exception as err:
        raise err
