from app.main.utils import logger


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']

    try:
        # TODO: call predictors and cetra
        res = "Received text from bot UI: " + utterance
        return res

    except Exception as err:
        raise err
