# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import pickle

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# input_text = "what is the capital of India?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))

# pickle.dump(model,open('flan_t5_large.pkl','wb'))

from transformers import T5Tokenizer
import pickle

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = pickle.load(open('flan_t5_large.pkl', 'rb'))

input_text = "what is the capital of India?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

