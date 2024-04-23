from huggingface_hub import notebook_login
notebook_login()
dataset_id = "samsum"
from datasets import load_dataset

# Load dataset from the hub
dataset = load_dataset(dataset_id)

# print(f"Train dataset size: {len(dataset['train'])}")
# print(f"Test dataset size: {len(dataset['test'])}")


print(dataset["train"][0:2])