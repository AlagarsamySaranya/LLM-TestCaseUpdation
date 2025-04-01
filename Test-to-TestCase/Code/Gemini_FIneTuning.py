import googleapiclient.discovery
import json
import time
import pandas as pd
from google.cloud import aiplatform

# Initialize Vertex AI client
aiplatform.init(project="your-project-id", location="us-central1")



# Define the project and location
project_id = '117340737262504381232'  
location = 'us-central1'

# Load the training and test datasets
train_file = "/home/saranya/HDD18TB/Test_TestCase/train.json"
test_file = "/home/saranya/HDD18TB/Test_TestCase/test.json"

df_train = pd.read_json(train_file)
df_test = pd.read_json(test_file)

# Define columns
old_method_col = "focal_src"   # Old method
new_method_col = "focal_tgt"   # New method
old_test_col = "test_src"      # Old test case
output_test_col = "test_tgt"   # New test case (expected output)

# Prepare training data in Gemini fine-tuning format
training_data = []
for _, row in df_train.iterrows():
    old_method = row[old_method_col]
    new_method = row[new_method_col]
    old_test_case = row[old_test_col]
    new_test_case = row[output_test_col]

    # Format as Gemini fine-tuning JSON format
    training_data.append({
        "input": f"### Old Method:\n{old_method}\n\n### New Method:\n{new_method}\n\n### Old Test Case:\n{old_test_case}\n\n### Generate New Test Case:",
        "output": new_test_case
    })

# Save formatted training data as JSONL (required for Gemini fine-tuning)
fine_tuning_file = "fine_tuning_data.jsonl"
with open(fine_tuning_file, "w") as f:
    for entry in training_data:
        f.write(json.dumps(entry) + "\n")

print(f"Fine-tuning data saved to {fine_tuning_file}")

# Upload the dataset to Google Cloud Storage (GCS)
gcs_bucket = "gs://your-bucket-name"
gcs_train_path = f"{gcs_bucket}/fine_tuning_data.jsonl"

aiplatform.gcs.upload_file(fine_tuning_file, gcs_train_path)

print(f"Training data uploaded to {gcs_train_path}")

# Create a fine-tuning job for Gemini
tuning_job = aiplatform.CustomJob(
    display_name="gemini-fine-tune-job",
    script_path="train_script.py",  # Custom script path for training
    container_uri="gcr.io/google.com/cloud/aiplatform/gemini-training",
    args=["--train_data", gcs_train_path]
)

tuning_job.run(sync=True)

print("Fine-tuning job submitted.")

# Wait for job completion
tuning_job.wait()

if tuning_job.state == "SUCCEEDED":
    print("Fine-tuning completed successfully.")
    model_id = tuning_job.resource_name
else:
    print("Fine-tuning failed.")
    exit()

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

# Perform inference using the fine-tuned Gemini model
responses = []
endpoint = aiplatform.Endpoint(model=model_id)

for test_instance in inference_data:
    prompt = f"### Old Method:\n{test_instance['old_method']}\n\n### New Method:\n{test_instance['new_method']}\n\n### Old Test Case:\n{test_instance['old_test_case']}\n\n### Generate New Test Case:"

    response = endpoint.predict(instances=[{"prompt": prompt}])

    generated_test_case = response.predictions[0] if response.predictions else "No response"
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
