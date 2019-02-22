import os
import unittest

from flask import current_app
from flask_testing import TestCase

from manage import app
from app.main.config import basedir


class TestDevConfig(TestCase):
    def create_app(self):
        app.config.from_object('app.main.config.DevelopmentConfig')

        return app

    def test_app_is_development(self):
        self.assertFalse(app.config['SECRET_KEY'] is 'secret_key')
        self.assertTrue(app.config['DEBUG'] is True)
        self.assertFalse(current_app is None)


class TestProductionConfig(TestCase):
    def create_app(self):
        app.config.from_object('app.main.config.ProductionConfig')

        return app

    def test_app_is_production(self):
        self.assertTrue(app.config['DEBUG'] is False)


if __name__ == '__main__':
    unittest.main()


