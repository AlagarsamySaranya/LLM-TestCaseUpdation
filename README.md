# Test-to-TestCase

- **Rule-based structural change detection**
- **LLM-based semantic analysis**

The approach identifies whether a test case requires an update when the corresponding product code changes and assists in generating the updated test case.

---

## 🚀 Datasets
test.json
train.json

Usage
1. Clone this repository
git clone https://github.com/AlagarsamySaranya/LLM-TestCaseUpdation.git
cd LLM-TestCaseUpdation

2. Set up OpenAI API Key
Create a file named openai_api_key.json in the root directory with your OpenAI API key:
{
    "api_key": "YOUR_OPENAI_API_KEY"
}

3. Run the detection pipeline
python Gpt.py
You can also run individual modules:
Replace Gpt.py with starCoder.py or Gemini.py


⚙️ Requirements
To install the required dependencies, run:
pip install -r requirements.txt

Rule-Based Analysis & Semantic Detection (LLM-Guided)
python Identify.py --input old_product.java --updated new_product.java -- old_test.java


