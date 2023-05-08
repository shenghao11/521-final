from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("../IR_qa_model")
model = AutoModelForSequenceClassification.from_pretrained("../IR_qa_model")
def represent(text):
  '''
  This function convert any text to a representation based on a fine-tuning model
  '''
  tokens = tokenizer.encode(text, add_special_tokens=True)
  input_ids = torch.tensor([tokens])
  outputs = model(input_ids)
  cls_embedding = outputs[0]
# Print the sentence embedding
  return(cls_embedding)
def similarity(text1_represent,text2_represent):
  '''
  This function calculate the cosine similarity between two tensors. The input should be two tensors.
  '''
  return torch.cosine_similarity(text1_represent,text2_represent)
data=pd.read_csv('../raw_data/database.csv')
data['Query_represent']=data['Query'].apply(lambda x: represent(x))
data.to_pickle('../clean_data/pre_encode_database.pkl')
