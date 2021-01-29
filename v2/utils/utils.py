import logging
import sys
import importlib
import json


def get_dict_str(dict_obj):
  message = ''
  message += '----------------- start ---------------\n'
  message += json.dumps(dict_obj, indent=2)
  message += '----------------- End -------------------'
  return message


def reload_module(module):
  if module not in sys.modules:
    imported_module = importlib.import_module(module)
  else:
    importlib.reload(sys.modules[module])


def register_modules(register_modules):
  for module in register_modules:
    reload_module(module=module)
    logging.getLogger('tl').info(f"  Register {module}")
  pass
