import ast
import streamlit as st


def parse_list_from_st_text_input(label, value):
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)

  parsed_value = ast.literal_eval(st_value)
  print(f"{label}: {value}")
  return parsed_value
