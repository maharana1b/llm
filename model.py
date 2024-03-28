from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "google/gemma-2b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set maximum length for generation
max_length = 500  # Adjust this value as needed

# Generate text with specified maximum length
input_text = "suggest 5 utterances like 'I want group life insurance'"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)