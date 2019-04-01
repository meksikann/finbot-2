from flask import request, jsonify
import traceback

from app.main.utils import logger
from app.main.controller import ga_request_handler


def init_routes(app):
    @app.route('/')
    def alive_route():
        logger.info('=========== Server alive 123===========>>>>>')
        return 'Server alive'

    # for slack testing
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

    # google assistant testing
    @app.route('/google_home/webhook', methods=['POST'])
    def handle_ga_connector_req():
        logger.info('Google Assistant connected to bot server')

        try:
            res = ga_request_handler.handle_qa_request(request.json)

            return jsonify(res), 200
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            return 'Error occurred while performing request', 500

