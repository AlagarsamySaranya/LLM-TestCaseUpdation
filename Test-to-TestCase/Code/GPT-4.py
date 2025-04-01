import openai
import json
import time
import pandas as pd

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

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

# Prepare training data in OpenAI fine-tuning format
training_data = []
for _, row in df_train.iterrows():
    old_method = row[old_method_col]
    new_method = row[new_method_col]
    old_test_case = row[old_test_col]
    new_test_case = row[output_test_col]

    # Format as OpenAI fine-tuning JSONL format
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a helpful AI that generates JUnit test cases."},
            {"role": "user", "content": f"### Old Method:\n{old_method}\n\n### New Method:\n{new_method}\n\n### Old Test Case:\n{old_test_case}\n\n### Generate New Test Case:"},
            {"role": "assistant", "content": new_test_case}
        ]
    })

# Save formatted training data as JSONL (required for OpenAI fine-tuning)
fine_tuning_file = "fine_tuning_data.jsonl"
with open(fine_tuning_file, "w") as f:
    for entry in training_data:
        f.write(json.dumps(entry) + "\n")

print(f"Fine-tuning data saved to {fine_tuning_file}")

# Upload the fine-tuning file to OpenAI
response = openai.File.create(
    file=open(fine_tuning_file, "rb"),
    purpose="fine-tune"
)

file_id = response["id"]
print(f"File uploaded successfully. File ID: {file_id}")

# Create a fine-tuning job
fine_tune_job = openai.FineTuningJob.create(
    model="gpt-4",  # Change to "gpt-3.5-turbo" if using GPT-3.5
    training_file=file_id
)

job_id = fine_tune_job["id"]
print(f"Fine-tuning job created. Job ID: {job_id}")

# Wait for the fine-tuning job to complete
while True:
    job_status = openai.FineTuningJob.retrieve(job_id)
    status = job_status["status"]

    if status == "succeeded":
        print("Fine-tuning completed successfully.")
        model_id = job_status["fine_tuned_model"]
        break
    elif status == "failed":
        print("Fine-tuning failed.")
        exit()
    else:
        print(f"Fine-tuning in progress... (Status: {status})")
        time.sleep(60)

print(f"Fine-tuned model ID: {model_id}")

# Load test data for inference
inference_data = []
for _, row in df_test.iterrows():
    old_method = row[old_method_col]
    new_method = row[new_method_col]
    old_test_case = row[old_test_col]

    inference_data.append({
        "old_method": old_method,
        "new_method": new_method,
        "old_test_case": old_test_case
    })

# Perform inference using the fine-tuned model
responses = []
for test_instance in inference_data:
    prompt = f"### Old Method:\n{test_instance['old_method']}\n\n### New Method:\n{test_instance['new_method']}\n\n### Old Test Case:\n{test_instance['old_test_case']}\n\n### Generate New Test Case:"

    response = openai.ChatCompletion.create(
        model=model_id,  # Use the fine-tuned model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    generated_test_case = response["choices"][0]["message"]["content"]
    responses.append({
        "old_method": test_instance["old_method"],
        "new_method": test_instance["new_method"],
        "old_test_case": test_instance["old_test_case"],
        "generated_test_case": generated_test_case
    })

# Save results to a file
output_file = "inference_results.json"
with open(output_file, "w") as f:
    json.dump(responses, f, indent=2)

print(f"Inference results saved to {output_file}")
