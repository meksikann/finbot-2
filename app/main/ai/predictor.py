from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from app.main.utils import logger
from app.main.ai import helper
from app.main import constants


def predict_intent(utterance):
    # import nlu settings

    domain_data = helper.get_domain_data()
    print(domain_data)


    # min_confidence = constants.THRASHOLD
    #
    # max_length = 10
    # glove_dimension = 100
    # logger.info('Start intent classification')
    # glove_path = constants.GLOVE_PATH
    # verbose = constants.VERBOSE
    #
    # # prepare test data ----------------------------------------------
    # word_index, classes = helper.get_token_data()
    #
    # tk = Tokenizer()
    #
    # tk.word_index = word_index
    #
    # x_test = tk.texts_to_sequences([utterance])
    # x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
    #
    # # get trained model ----------------------------------------------
    # glove_dict = helper.generate_glove_dict(glove_path)
    # vocab_size = len(tk.word_index) + 1
    # embed_matrix = helper.get_embedding_matrix(glove_dict, tk.word_index.items(), vocab_size, glove_dimension)
    # model = helper.get_glove_model(vocab_size, glove_dimension, embed_matrix, max_length, len(classes))
    #
    # model = helper.load_model_weights(model)
    #
    # print('x_test', x_test)
    #
    # # predict --------------------------------------------------------
    # prediction = model.predict(x_test, verbose=verbose)
    #
    # predicted_class = helper.get_predicted_class(min_confidence, prediction, classes)
    # print('Predicted intent is:', predicted_class)
    #
    # return predicted_class