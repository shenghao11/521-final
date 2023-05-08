import streamlit as st
from streamlit_chat import message

from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("./IR_qa_model")
model = AutoModelForSequenceClassification.from_pretrained("./IR_qa_model")

def represent(text):
  '''
  This function convert any text to a representation based on a fine-tuning model
  '''
  tokens = tokenizer.encode(text, add_special_tokens=True)
  input_ids = torch.tensor([tokens])
  outputs = model(input_ids)
  cls_embedding = outputs[0]
  return(cls_embedding)

def similarity(text1_represent,text2_represent):
  '''
  This function calculate the cosine similarity between two tensors. The input should be two tensors.
  '''
  return torch.cosine_similarity(text1_represent,text2_represent)

data=pd.read_pickle('./data/clean_data/pre_encode_database.pkl')


def generate_response(prompt):
  '''
  This function call the represent function, convert the question to the embedding. 
  It then apply the similarity function to all questions in the database, and return the answer of that has the highest score.
  '''
  question=represent(prompt)
  data['temporary']=data['Query_represent'].apply(lambda x: similarity(x,question))
  return data[data['temporary']==data['temporary'].max()]['Key'].values[0]

st.title("NLP Question Answering Bot")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input=st.text_input("You:",key='input')
if user_input:
    output=generate_response(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
  
