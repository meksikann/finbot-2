from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

from app.main.utils import logger
from app.main.ai import helper
from app.main import constants


def predict_intent(utterance):
    # import nlu settings

    min_confidence = constants.THRASHOLD

    max_length = 10
    glove_dimension = 100
    logger.info('Start intent classification')
    glove_path = constants.GLOVE_PATH
    verbose = constants.VERBOSE

    # prepare test data ----------------------------------------------
    word_index, classes, embed_matrix = helper.get_token_data()

    tk = Tokenizer()

    tk.word_index = word_index

    x_test = tk.texts_to_sequences([utterance])
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

    # get trained model ----------------------------------------------
    # glove_dict = helper.generate_glove_dict(glove_path)
    vocab_size = len(tk.word_index) + 1
    # embed_matrix = helper.get_embedding_matrix(glove_dict, tk.word_index.items(), vocab_size, glove_dimension)
    model = helper.get_glove_model(vocab_size, glove_dimension, embed_matrix, max_length, len(classes))

    model = helper.load_nlp_model_weights(model)
    # predict --------------------------------------------------------
    prediction = model.predict(x_test, verbose=verbose)

    predicted_class = helper.get_predicted_class(min_confidence, prediction, classes)

    return predicted_class


def predict_action(domain_tokens, maxlen, num_features, sequence):
    logger.info('================>>>>>>>>>>>>>>>> START DIALOG FLOW PREDICTION <<<<<<<<<<<<<<<<<=============')
    try:
        min_confidence = constants.THRASHOLD
        # prepare test data
        x_test = np.array(sequence)
        x_test = pad_sequences([x_test],  maxlen=maxlen, padding='post')
        # x_test = np.reshape(x_test, (1, x_test.shape[1], num_features))

        # get model
        model = helper.create_dialog_network(maxlen, len(domain_tokens))

        # load weights
        model = helper.load_dm_model_weights(model)

        # reset model states
        # model.reset_states()
        print('X_test: ', x_test)

        pred = model.predict(x_test, verbose=1)
        # predicted_class = helper.get_predicted_class(min_confidence, pred, domain_tokens)
        max_score_index = np.argmax(pred)


        # # get array with domain_tokens values
        # tokens = []
        for key, val in domain_tokens.items():
            if val == max_score_index:
                class_predicted = key
                break
        print('predicted------->>>>>>>', class_predicted)
        # # get closer value form list
        # pred = helper.get_closes_value(tokens, pred[0][0])
        return max_score_index
    except Exception as err:
        logger.error(err)
        raise err


def restart_predictor():
    helper.clear_prediction_data()
