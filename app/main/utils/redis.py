import redis
from app.main.utils import logger
import pickle


redis_host = 'localhost'
redis_port = 6379
redis_pass = ''

r = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    password=redis_pass
)


def set_data(key, value):
    try:
        data = pickle.dumps(value)
        r.set(key, data)
        logger.info('Data has been set to Radis')

    except Exception as err:
        logger.error(err)
        return None


def get_data(key):
    try:
        logger.info('Data has been got from Radis')

        return pickle.loads(r.get(key))

    except Exception as err:
        logger.error(err)
        return None

