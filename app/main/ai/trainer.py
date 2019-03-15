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


def split_to_sequences(sequence):
    x, y = [], []
    max_length = len(sequence) - 1

    for idx, seq in enumerate(sequence):
        x_sample, y_sample = [], []
        # stop when last sample treated
        if idx == max_length:
            break

        x_sample = sequence[: idx+1]

        x.append(x_sample)
        y.append(sequence[idx+1])

    return x, y


def train_dialog_model():
    logger.info('Start train dialog model ----------->>>>>>>>')
    domain_tokens = dict()
    training_data = []
    x_train = []
    y_train = []
    max_length = 1
    num_features = 1
    time_steps = 1
    num_epochs = 300

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

        # tokenize dialogs
        for flow in dialog_data['dialogs']:
            sequence = flow['flow']
            sequence = list(map(lambda sq: domain_tokens[sq], sequence))

            # get max list length
            if len(sequence) > max_length:
                max_length = len(sequence)

            training_data.append(sequence)

        # get dialogs flow versions
        for sample in training_data:
            splited_seq, labels = split_to_sequences(sample)
            x_train = x_train + splited_seq
            y_train = y_train + labels

        # pad sequences
        x_train = pad_sequences(x_train, maxlen=max_length-1, padding='post')

        print(x_train)
        print(y_train)

        # reshape X to proper dimension [samples, timestamps ,features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))

        # get LSTM model --------------------------------------------------------------------------->>>>>>>>>>>>
        model = helper.create_dialog_network(max_length - 1, num_features)
        model.compile(loss='mse', optimizer='adam')

        model.summary()

        # fit model
        model.fit(x_train, y_train, epochs=num_epochs, batch_size=num_features, verbose=1)

        # save model
        helper.save_dialog_model(model)

        # save dialog tokens
        helper.save_dialog_options(domain_tokens, num_features, sample_length=max_length-1)

        print('================>>>>>>>>>>>>>>>>DIALOG TRAINING DONE<<<<<<<<<<<<<<<<<=============')
        # print('================>>>>>>>>>>>>>>>> START TEST CASE <<<<<<<<<<<<<<<<<=============')
        # x_test = np.array([9])
        # x_test = pad_sequences([x_test],  maxlen=max_length-1, padding='post')
        # x_test = np.reshape(x_test, (1, x_test.shape[1], num_features))
        #
        # pred = model.predict(x_test, verbose=1)
        # print('TEST CASE RESULT:', pred)

    except Exception as err:
        raise err
