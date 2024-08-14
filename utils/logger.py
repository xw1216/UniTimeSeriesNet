import os
import logging


def check_file_handler(logger: logging.Logger, path):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return h.baseFilename == os.path.abspath(path)
    return False


def check_stream_handler(logger: logging.Logger):
    for h in logger.handlers:
        if not isinstance(h, logging.FileHandler) and isinstance(h, logging.StreamHandler):
            return True
    return False


def get_logger(
        name='log',
        path=None,
        level=logging.INFO
):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

    log = logging.getLogger(name)

    if path is not None and not check_file_handler(log, path):
        log_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(log_formatter)
        log.addHandler(file_handler)

    if not check_stream_handler(log):
        log_formatter = logging.Formatter(
            fmt='%(asctime)s \x1b[36m[%(levelname)-7s]\x1b[0m %(message)s',
            datefmt="%H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        log.addHandler(console_handler)
        log.setLevel(level)

    return log


if __name__ == '__main__':
    log = get_logger('test', os.path.join(os.pardir, 'test', 'test.log'))
    log.debug('debug')
    log.info('info')
    log.warning('warn')
    log.error('error')
