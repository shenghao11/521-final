from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("./IR_qa_model")
model = AutoModelForSequenceClassification.from_pretrained("./IR_qa_model")
tex1='give me some information about stopwords'
tex2='what are stopwords'
tex3='what is a language model'
def represent(text):
  tokens = tokenizer.encode(text, add_special_tokens=True)
  input_ids = torch.tensor([tokens])
  outputs = model(input_ids)
  cls_embedding = outputs[0]
# Print the sentence embedding
  return(cls_embedding)
def similarity(text1,text2):
  return torch.cosine_similarity(represent(text1),represent(text2))
print(similarity(tex1,tex2))
print(similarity(tex1,tex3))