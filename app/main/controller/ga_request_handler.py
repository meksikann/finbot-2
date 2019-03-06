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
        prediction = predictor.predict_intent(utterance)
        if prediction is not None:
            print(prediction)

        # TODO: get next bot action

        # TODO: if action utterance - send bot response

        return result

    except Exception as err:
        raise err
