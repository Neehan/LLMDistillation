import logging


class TqdmToLogger(object):
    """File-like object to redirect tqdm output to logging."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        self.buf += buf
        if "\n" in self.buf:
            self.flush()
            self.buf = ""

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf.strip("\r\n\t "))
