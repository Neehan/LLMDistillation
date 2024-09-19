import logging


class TqdmToLogger(object):
    """File-like object to redirect tqdm output to logging."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        self.buf += buf
        while "\n" in self.buf or "\r" in self.buf:
            if "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
            else:
                line, self.buf = self.buf.split("\r", 1)
            self.logger.log(self.level, line.strip("\r\n\t "))

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf.strip("\r\n\t "))
            self.buf = ""
