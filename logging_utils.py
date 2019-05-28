import logging
import datetime

FORMAT = "%(message)s [%(name)s %(levelname)s][%(filename)s:%(funcName)s():%(lineno)s][%(asctime)s.%(msecs)03d]"
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


def get_logger(filename, propagate=True, level=logging.INFO, stream=False):
  """

  :param filename:
  :param propagate: whether log to stdout
  :return:
  """
  import logging
  logger = logging.getLogger(name=filename)
  logger.setLevel(level)
  logger.propagate = propagate

  formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

  file_hander = logging.FileHandler(filename=filename, mode='w')
  file_hander.setLevel(level=level)
  file_hander.setFormatter(formatter)
  logger.addHandler(file_hander)

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

  if stream:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

  return logger
