import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Load dataset safely (Fix JSON decoding issues)
train_file = "train.json"
test_file = "test.json"

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)

df_train = pd.DataFrame(load_json_file(train_file))
df_test = pd.DataFrame(load_json_file(test_file))

# Define input and output columns
old_method_col = "focal_src"   # Old method
new_method_col = "focal_tgt"   # New method
old_test_col = "test_src"      # Old test case
output_test_col = "test_tgt"   # New test case (expected output)

# Load InCoder model and tokenizer
model_name = "facebook/incoder-6B"  # You can replace this with a smaller model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Prepare training data in Hugging Face Dataset format
train_data = []
for _, row in df_train.iterrows():
    old_method = row[old_method_col]
    new_method = row[new_method_col]
    old_test_case = row[old_test_col]
    new_test_case = row[output_test_col]  # Expected new test case

    # Format input-output as prompt-completion pair (InCoder uses special tokens for completion)
    train_data.append({
        "text": f"<|file|>\nOld Method:\n{old_method}\n\nNew Method:\n{new_method}\n\nOld Test Case:\n{old_test_case}\n\nGenerate New Test Case:\n{new_test_case}<|endoftext|>"
    })

train_dataset = Dataset.from_dict({"text": [item["text"] for item in train_data]})

# Define training arguments
training_args = TrainingArguments(
    output_dir="./incoder_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs"
)

# ✅ Corrected Trainer definition
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset  # ✅ Fixed the typo
)

# Train model
print("Starting fine-tuning of InCoder...")
trainer.train()
print("Fine-tuning complete.")

# Save the fine-tuned model
trainer.save_model("./incoder_finetuned")
print("Model saved.")

# Load fine-tuned model for inference
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./incoder_finetuned")
fine_tuned_model.to("cuda")

# Function to generate new test cases using InCoder
def generate_test_case(old_method, new_method, old_test_case):
    input_text = f"<|file|>\nOld Method:\n{old_method}\n\nNew Method:\n{new_method}\n\nOld Test Case:\n{old_test_case}\n\nGenerate New Test Case:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = fine_tuned_model.generate(**inputs, max_length=300, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Perform inference on test data
results = []
for _, row in df_test.iterrows():
    old_method = row[old_method_col]
    new_method = row[new_method_col]
    old_test_case = row[old_test_col]

    generated_test_case = generate_test_case(old_method, new_method, old_test_case)
    
    results.append({
        "old_method": old_method,
        "new_method": new_method,
        "old_test_case": old_test_case,
        "generated_test_case": generated_test_case
    })

# Save results
output_file = "inference_results_incoder.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Inference results saved to {output_file}")
