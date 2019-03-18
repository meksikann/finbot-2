from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']
    userID = '1234' # connector ID/userId should be used here to store user dialog state

    try:
        result = ""

        # get user intent
        # prediction = predictor.predict_intent(utterance)
        # if prediction is not None:
        #     print(prediction)

        domain_tokens, maxlen, num_features = helper.get_dialog_options()
        state = helper.get_dialog_state()
        # TODO: generate x_test for DM prediction/ restore dialog state
        x_test = [8, 17, 9]

        # predict next action
        action_predicted = predictor.predict_action(domain_tokens, maxlen, num_features, x_test)
        logger.info('PREDICTED ACTION: {}'.format(action_predicted))

        # TODO: if action utterance - send bot response if Action do Action  and save dialog STATE , elif NONE  -
        #  respond with utter_repeat_again  and DO NOT save Dialog state
        # save chat state

        return result

    except Exception as err:
        raise err
