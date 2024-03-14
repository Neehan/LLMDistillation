import logging


class TqdmToLogger(object):
    """File-like object to redirect tqdm output to logging."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)
