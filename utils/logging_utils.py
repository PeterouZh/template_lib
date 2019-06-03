import logging
import datetime

FORMAT = "[%(levelname)s]: %(message)s [%(name)s][%(filename)s:%(funcName)s():%(lineno)s][%(asctime)s.%(msecs)03d]"
DATEFMT = '%Y/%m/%d %H:%M:%S'


def logging_init(filename=None, level=logging.INFO, correct_time=False):
  def beijing(sec, what):
    '''sec and what is unused.'''
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

  if correct_time:
    logging.Formatter.converter = beijing

  logging.basicConfig(level=level,
                      format=FORMAT,
                      datefmt=DATEFMT,
                      filename=None, filemode='w')
  logger = logging.getLogger()

  # consoleHandler = logging.StreamHandler()
  # logger.addHandler(consoleHandler)

  if filename:
    logger_handler = logging.FileHandler(filename=filename, mode='w')
    logger_handler.setLevel(level=level)
    logger_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    logger.addHandler(logger_handler)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg
  return logger

  # logging.error('There are something wrong', exc_info=True)


def get_root_logger(filename, stream=True, level=logging.INFO):
  logger = logging.getLogger()
  logger.setLevel(level)
  set_hander(logger=logger, filename=filename, stream=stream, level=level)
  return logger


def get_logger(filename, stream=True, level=logging.INFO):
  """

  :param filename:
  :param propagate: whether log to stdout
  :return:
  """
  logger = logging.getLogger(filename)
  logger.setLevel(level)
  set_hander(logger=logger, filename=filename, stream=stream, level=level)
  return logger


def set_hander(logger, filename, stream=True, level=logging.INFO):
  formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

  file_hander = logging.FileHandler(filename=filename, mode='a')
  file_hander.setLevel(level=level)
  file_hander.setFormatter(formatter)
  logger.addHandler(file_hander)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("  %(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg

  if stream:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

  logger.propagate = False
  return logger
