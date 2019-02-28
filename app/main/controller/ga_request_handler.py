from app.main.utils import logger
from app.main.ai import predictor


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']

    try:
        # TODO: call predictors and cetra
        result = "Received text from bot UI: "

        prediction = predictor.predict_intent(utterance)

        if prediction is not None:
            print(prediction)

        return result

    except Exception as err:
        raise err
