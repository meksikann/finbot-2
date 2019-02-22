from flask import request
import traceback

from app.main.utils import logger
from app.main.controller import ga_request_handler


def init_routes(app):
    @app.route('/')
    def alive_route():
        logger.info('=========== Server alive 123===========>>>>>')
        return 'Server alive'

    @app.route('/test-route', methods=['POST'])
    def test_route():
        logger.info('=========== test routes ===========>>>>>')
        try:
            res = ga_request_handler.handle_qa_request(request.json)

            return res, 200
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            return 'Error occurred while performing request', 500

    @app.route('/ga-connector', methods=['POST'])
    def handle_ga_connector_req():
        logger.info('Google Assistant connected to bot server')

        try:
            res = ga_request_handler.handle_qa_request(request.json)

            return res, 200
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            return 'Error occurred while performing request', 500

