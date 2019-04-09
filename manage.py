import os
import unittest
# from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from app.main import create_app
from app.main.utils import logger
from app.main.ai import trainer
from app.main.ai import predictor
from app.main.controller import starter

app = create_app(os.getenv('CUSTOM_ENV') or 'dev')

# to use flask context globals
app.app_context().push()

manager = Manager(app)


@manager.command
def run():
    app.run(host='0.0.0.0', port=os.getenv('PORT') or 8282)


@manager.command
def test():
    """run unit tests"""
    tests = unittest.TestLoader().discover('app/tests', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)

    if result.wasSuccessful():
        return 0
    return 1


@manager.command
def train_nlu():
    """ start train NN responsible for intent classification"""

    logger.info('Command: train_nlu')
    trainer.train_intent_model()


@manager.command
def train_dialog():
    """ start train NN responsible for dialog actions prediction"""
    logger.info('Command: train_dialog')
    trainer.train_dialog_model()


@manager.command
def restart_predictor():
    """ used to remove all user dialog states (to start from total beginning)"""

    logger.info('Command: restart predictor: remove dialog_state.pkl')
    predictor.restart_predictor()


@manager.command
def start_desktop_voice_assistant():
    """Desktop voice assistant to uses machine mic"""
    logger.info('Starting desktop voice assistant ...')
    starter.start_desktop_voice_assistant()


if __name__ == '__main__':
    manager.run()
