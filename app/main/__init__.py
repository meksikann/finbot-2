from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS

from .config import config_by_name
from .routes import routes

flask_bcrypt = Bcrypt()


def create_app(config_name):
    """Create app - Factory pattern"""
    app = Flask(__name__)
    # create app instance
    app.config.from_object(config_by_name[config_name])
    flask_bcrypt.init_app(app)

    CORS(app)

    routes.init_routes(app)

    return app
