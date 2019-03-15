from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']

    try:
        result = ""

        # get user intent
        # prediction = predictor.predict_intent(utterance)
        # if prediction is not None:
        #     print(prediction)

        # TODO: generate x_test for DM prediction

        # TODO: get next bot action
        action_predicted = predictor.predict_action([8, 17, 9])  # wait for 16
        print('PREDICTED ACTION: ', action_predicted)

        # TODO: if action utterance - send bot response

        # save chat state

        return result

    except Exception as err:
        raise err
