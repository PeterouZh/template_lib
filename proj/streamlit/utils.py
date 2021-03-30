import logging
import json
import ast
import streamlit as st
import pandas as pd

from template_lib.utils import read_image_list_from_files
from . import SessionState

def is_init():
  try:
    saved_suffix_state = SessionState.get(saved_suffix=0)
  except:
    return False
  return True

class LineChart(object):
  def __init__(self, x_label, y_label):
    self.x_label = x_label
    self.y_label = y_label

    self.pd_data = pd.DataFrame(columns=[x_label, y_label])
    self.st_line_chart = st.empty()
    self.st_init = is_init()
    pass

  def write(self, x, y):
    if not self.st_init:
      return
    self.pd_data = self.pd_data.append({self.x_label: x, self.y_label: y}, ignore_index=True)
    pd_data = self.pd_data.set_index(self.x_label)
    self.st_line_chart.line_chart(pd_data)
    pass



def selectbox(label, options, index=0):
  ret = st.selectbox(label=label, options=options, index=index)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret

def number_input(label,
                 value,
                 min_value=None,
                 step=None,
                 format=None, # "%.8f"
                 **kwargs):
  st_empty = st.empty()
  ret = st_empty.number_input(label=f"{label}: {value}", value=value, min_value=min_value,
                              step=step, format=format, **kwargs)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def checkbox(label,
             value):
  st_empty = st.empty()
  ret = st_empty.checkbox(label=f"{label}: {value}", value=value)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def text_input(label,
               value,
               **kwargs):
  ret = st.text_input(label=f"{label}: {value}", value=value, key=label)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def parse_list_from_st_text_input(label, value):
  """
  return: list
  """
  value = str(value)
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)

  parsed_value = ast.literal_eval(st_value)
  print(f"{label}: {parsed_value}")
  logging.getLogger('st').info(f"label: {parsed_value}")
  return parsed_value


def read_image_list_and_show_in_st(image_list_file, columns=['path', 'class_id']):
  if not isinstance(image_list_file, (list, tuple)):
    image_list_file = [image_list_file, ]

  st.header("Image list file: ")
  for image_file in image_list_file:
    st.write(image_file)

  all_image_list = read_image_list_from_files(image_list_file)
  image_list_df = pd.DataFrame(all_image_list, columns=columns)
  st.dataframe(image_list_df)
  return all_image_list


def parse_dict_from_st_text_input(label, value):
  if isinstance(value, dict):
    value = json.dumps(value)
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)
  parse_value = json.loads(st_value)
  print(f"{label}: {parse_value}")
  logging.getLogger('st').info(f"label: {parse_value}")
  return parse_value











