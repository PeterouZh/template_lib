import logging
import json
import ast
import streamlit as st
import pandas as pd

from template_lib.utils import read_image_list_from_files


def parse_list_from_st_text_input(label, value):
  """
  return: list
  """
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)

  parsed_value = ast.literal_eval(st_value)
  print(f"{label}: {parsed_value}")
  logging.getLogger('st').info(f"label: {parsed_value}")
  return parsed_value


def read_image_list_and_show_in_st(image_list_file, columns=['path', 'class_id']):
  all_image_list = read_image_list_from_files(image_list_file)

  image_list_df = pd.DataFrame(all_image_list, columns=columns)
  header = ",\n ".join(image_list_file)
  st.write(header)
  st.dataframe(image_list_df)
  print(f"Loaded dataframe from {header}")
  return all_image_list


def parse_dict_from_st_text_input(label, value):
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)
  parse_value = json.loads(st_value)
  print(f"{label}: {parse_value}")
  logging.getLogger('st').info(f"label: {parse_value}")
  return parse_value










