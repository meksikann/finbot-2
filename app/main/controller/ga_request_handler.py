from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper
from flask import jsonify


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    # logger.info(data)

    utterance = ''
    is_slack_channel = False
    channel_id = None
    user_id = '1234'  # connector ID/userId should be used here to store user dialog state TODO: mreceive userId from connector

    # *******************************  slack connection ***********************************************************
    # slack event url testing
    if helper.check_key_exists(data, 'type') and data['type'] == 'url_verification':
        return data['challenge']

    # receive user event from slack ------------>>>>>>>>>>
    if helper.check_key_exists(data, 'event'):

        # do not handle request if bot message event occurred
        if helper.check_key_exists(data['event'], 'subtype'):
            return ''
        # get data from user message
        if data['event']['type'] == 'message':
            utterance = data['event']['text']
            user_id = data['event']['user']
            channel_id = data['event']['channel']
            is_slack_channel = True
    # receive event from postman ------------>>>>>>>>>>
    else:
        utterance = data['utterance']

    # *************************************************************************************************************

    try:
        text_response = ""
        domain_tokens, maxlen, num_features = helper.get_dialog_options()

        print(domain_tokens)
        # get user intent
        logger.info('User said: ' + utterance)
        prediction = predictor.predict_intent(utterance)
        logger.info('Predicted intent token:{}'.format(prediction))

        if prediction is not None:
            utterance_token = domain_tokens[prediction]
        else:
            text_response = helper.generate_utter('utter_repeat_again')

            # post slack message
            if is_slack_channel:
                helper.post_slack_message(text_response, channel_id)
            return text_response

        # get dialog state
        state = helper.get_dialog_state()

        if state is not None:
            pass
        else:
            state = dict({user_id: []})

        # generate x_test for DM prediction/ restore dialog state
        x_test = state[user_id]
        # update user dialog state with new utterance
        x_test.append(utterance_token)

        # get lats n-actions with dialog length dimension
        if len(x_test) > maxlen:
            x_test = x_test[-maxlen:]

        # predict next action
        print('actions before predict:', x_test)
        action_predicted = predictor.predict_action(domain_tokens, maxlen, num_features, x_test)

        if action_predicted is not None:
            # save max length+1 actions
            x_test.append(action_predicted)

        logger.info('PREDICTED ACTION: {}'.format(action_predicted))
        text_response = helper.get_utterance(domain_tokens, action_predicted)

        print('actions after predict:', x_test)
        # save chat state
        state[user_id] = x_test
        helper.save_dialog_state(state)

        # post slack message
        if is_slack_channel:
            helper.post_slack_message(text_response, channel_id)

        return text_response

    except Exception as err:
        raise err
