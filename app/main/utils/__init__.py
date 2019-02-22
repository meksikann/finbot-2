import logzero
from logzero import logger


logzero.logfile("logs/logs.log", maxBytes=1e6, backupCount=3)
