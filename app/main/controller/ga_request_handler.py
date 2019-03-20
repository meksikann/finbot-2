from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper
from app.main import constants


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""

    logger.info('Got request from GA.')
    # logger.info(data)

    proceed, utterance, user_id, channel_id, channel_name = get_data_form_channel(data)

    # if no need to invoke predictor
    if proceed is False:
        return ''

    next_user_action = process_next_user_action(utterance, user_id)

    # post slack message
    if channel_name == constants.SLACK:
        helper.post_slack_message(next_user_action['text'], channel_id)
        return ''

    return next_user_action['text']


def get_data_form_channel(req):
    """define the channel and extract data for predictor"""
    print(req)
    utterance = ''
    channel_name = None
    channel_id = None
    user_id = None
    proceed = False

    # *******************************  slack connection ***********************************************************
    # slack event url testing
    if helper.check_key_exists(req, 'type') and req['type'] == 'url_verification':
        utterance = req['challenge']
        proceed = False
    else:
        # receive user event from slack ------------>>>>>>>>>>
        if helper.check_key_exists(req, 'event'):

            # do not handle request if bot message event occurred
            if helper.check_key_exists(req['event'], 'subtype') and req.get('event', {}).get('subtype',
                                                                                             None) == 'bot_message':
                utterance = ''
                proceed = False
            else:
                # get data from user message
                if req['event']['type'] == 'message':
                    utterance = req['event']['text']
                    user_id = req['event']['user']
                    channel_id = req['event']['channel']
                    channel_name = constants.SLACK
                    proceed = True
        # receive event from postman ------------>>>>>>>>>>
        else:
            proceed = True
            utterance = req['utterance']

    return proceed, utterance, user_id, channel_id, channel_name


def process_next_user_action(utterance, user_id='1234'):
    """user AI predictor to generate next bot response/action"""

    bot_response = dict()

    try:
        domain_tokens, maxlen, num_features = helper.get_dialog_options()

        print(domain_tokens)
        # get user intent
        logger.info('User said: ' + utterance)
        prediction = predictor.predict_intent(utterance)
        logger.info('Predicted intent token:{}'.format(prediction))

        if prediction is None:
            text_response = helper.generate_utter('utter_repeat_again')
        else:
            utterance_token = domain_tokens[prediction]

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

            logger.info(
                'PREDICTED ACTION: {} with index {}'.format(get_key(domain_tokens, action_predicted), action_predicted))
            text_response = helper.get_utterance(domain_tokens, action_predicted)

            print('actions after predict:', x_test)
            # save chat state
            state[user_id] = x_test
            helper.save_dialog_state(state)

        bot_response['text'] = text_response

        return bot_response
    except Exception as err:
        raise err


def get_key(dct, val):
    for key, value in dct.items():
        if value == val:
            return key
