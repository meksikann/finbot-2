from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']

    try:
        # TODO: call predictors and cetra
        result = "Received text from bot UI: "

        # prediction = predictor.predict_intent(utterance)
        # if prediction is not None:
        #     print(prediction)

        domain_data = helper.get_domain_data()
        print('Domain data:', domain_data['actions_list'])

        # dialog_data = helper.get_dialog_flow_data()
        # print('Dialog data: ', dialog_data['dialogs'])



        return result

    except Exception as err:
        raise err
