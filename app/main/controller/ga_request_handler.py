from app.main.utils import logger
from app.main.ai import predictor
from app.main.ai import helper


def handle_qa_request(data):
    """handle request from Google assistant: call prediction services to generate next bot response"""
    logger.info('Got request from GA.')
    logger.info(data)

    utterance = data['utterance']
    userID = '1234' # connector ID/userId should be used here to store user dialog state TODO: mreceive userId from connector

    try:
        result = ""
        domain_tokens, maxlen, num_features = helper.get_dialog_options()

        print(domain_tokens)
        #get user intent
        logger.info('User said: ' + utterance)
        prediction = predictor.predict_intent(utterance)
        logger.info('Predicted intent token:{}'.format(prediction))

        if prediction is not None:
            utterance_token = domain_tokens[prediction]
        else:
            result = helper.generate_utter('utter_repeat_again')
            return result

        # get dialog state
        state = helper.get_dialog_state()

        if state is not None:
            pass
        else:
            state = dict({userID: []})

        # generate x_test for DM prediction/ restore dialog state
        x_test = state[userID]
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
        result = helper.get_utterance(domain_tokens, action_predicted)

        print('actions after predict:', x_test)
        # save chat state
        state[userID] = x_test
        helper.save_dialog_state(state)

        return result

    except Exception as err:
        raise err
