import javalang
import difflib
import os
import pandas as pd
from openai import OpenAI
import time

# Initialize OpenAI Client
client = OpenAI(api_key="")
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def get_method_signatures(java_code):
    try:
        tree = javalang.parse.parse(java_code)
    except javalang.parser.JavaSyntaxError:
        return {}

    methods = {}
    for _, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            methods[node.name] = {
                'return_type': str(node.return_type) if node.return_type else "void",
                'exceptions': [str(e.name) for e in node.throws] if node.throws else [],
                'params': [(p.name, str(p.type)) for p in node.parameters]
            }
    return methods


def extract_test_info(test_code):
    try:
        tree = javalang.parse.parse(test_code)
    except javalang.parser.JavaSyntaxError:
        return {'assertions': [], 'expected_exceptions': [], 'method_calls': []}

    assertions, expected_exceptions, method_calls = [], [], []

    for _, node in tree:
        if isinstance(node, javalang.tree.MethodInvocation):
            method_calls.append(node.member)
            if node.member.startswith("assert"):
                assertions.append({
                    'type': node.member,
                    'arguments': [str(arg) for arg in node.arguments]
                })
        if isinstance(node, javalang.tree.Annotation):
            if 'expected' in str(node):
                expected_exceptions.append(str(node))

    return {'assertions': assertions, 'expected_exceptions': expected_exceptions, 'method_calls': method_calls}


def rule_based_detection(old_code, new_code):
    old_methods = get_method_signatures(old_code)
    new_methods = get_method_signatures(new_code)

    reasons = []
    for method in old_methods:
        if method in new_methods:
            if old_methods[method]['return_type'] != new_methods[method]['return_type']:
                reasons.append(f"Return type of '{method}' changed")
            if old_methods[method]['params'] != new_methods[method]['params']:
                reasons.append(f"Parameters of '{method}' changed")
            if set(old_methods[method]['exceptions']) != set(new_methods[method]['exceptions']):
                reasons.append(f"Throws clause of '{method}' changed")

    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    significant_changes = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
    if significant_changes:
        reasons.append("Significant control flow changes detected")

    return reasons


def analyze_test_against_new_code(test_code, new_code):
    test_info = extract_test_info(test_code)
    new_methods = get_method_signatures(new_code)

    analysis = {
        'method_calls_analysis': [],
        'assertion_analysis': [],
        'expected_exceptions': [],
        'detailed_text': ""
    }

    detailed_lines = []

    for call in test_info['method_calls']:
        if call in new_methods:
            detailed_lines.append(f"✔️ Method call '{call}' exists in the new product code.\n")
            analysis['method_calls_analysis'].append(f"'{call}' exists in new code")
        else:
            detailed_lines.append(f"❗ Method call '{call}' is missing or renamed in the new product code.\n")
            analysis['method_calls_analysis'].append(f"'{call}' missing or renamed in new code")

    for assertion in test_info['assertions']:
        involved_methods = [m for m in new_methods if m in assertion['arguments']]
        if involved_methods:
            for method in involved_methods:
                return_type = new_methods[method]['return_type']
                detailed_lines.append(
                    f"✔️ Assertion '{assertion['type']}' involves method '{method}' which returns '{return_type}'.\n"
                )
                analysis['assertion_analysis'].append(
                    f"Assertion '{assertion['type']}' involves method '{method}' with return type '{return_type}'"
                )
        else:
            detailed_lines.append(
                f"❗ Assertion '{assertion['type']}' arguments do not match any method in the new product code.\n"
            )

    if test_info['expected_exceptions']:
        detailed_lines.append(f"ℹ️ Test case has expected exceptions: {test_info['expected_exceptions']}\n")
        analysis['expected_exceptions'] = test_info['expected_exceptions']
    else:
        detailed_lines.append(f"ℹ️ No expected exceptions specified in the test case.\n")

    analysis['detailed_text'] = "".join(detailed_lines)
    return analysis


def llm_semantic_detection_guided(old_code, new_code, test_code, rule_reasons):
    test_analysis = analyze_test_against_new_code(test_code, new_code)

    summary_lines = []
    if test_analysis['method_calls_analysis']:
        summary_lines.extend(
            [f"- {line}" for line in test_analysis['method_calls_analysis']]
        )
    else:
        summary_lines.append("- No method calls found in test case.")

    if test_analysis['assertion_analysis']:
        summary_lines.extend(
            [f"- {line}" for line in test_analysis['assertion_analysis']]
        )
    else:
        summary_lines.append("- No valid assertion matches any method in the new product code.")

    if test_analysis['expected_exceptions']:
        summary_lines.append(f"- Test case has expected exceptions: {test_analysis['expected_exceptions']}")
    else:
        summary_lines.append("- No expected exceptions specified in the test case.")

    summary_text = "\n".join(summary_lines)

    prompt = f"""
You are an expert software developer specialising in code maintenance and testing. 
Determine if the test case requires an update based on the following analysis:

---  Code Changes ---
{rule_reasons if rule_reasons else "None"}

--- Test-to-New-Code Analysis Summary ---
{summary_text}

--- Old Product Code ---
{old_code}

--- New Product Code ---
{new_code}

--- Test Case ---
{test_code}

Reply exactly with:
- "Needs Update: Yes" OR "Needs Update: No"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content.strip()
        return reply, summary_text
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return None, None


# Main execution
base_path = "/home/saranya/HDD18TB/Test_TestCase/sampletest"
records = []

for project in os.listdir(base_path):
    project_path = os.path.join(base_path, project)
    if os.path.isdir(project_path):
        for commit_id in os.listdir(project_path):
            commit_path = os.path.join(project_path, commit_id)
            if os.path.isdir(commit_path):
                for folder in os.listdir(commit_path):
                    folder_path = os.path.join(commit_path, folder)
                    try:
                        old_method_code = read_file(os.path.join(folder_path, "old_product.java"))
                        new_method_code = read_file(os.path.join(folder_path, "new_product.java"))
                        old_test_code = read_file(os.path.join(folder_path, "old_test.java"))
                        label = read_file(os.path.join(folder_path, "label.txt")).strip()

                        rule_reasons = rule_based_detection(old_method_code, new_method_code)

                        final_label, reason = llm_semantic_detection_guided(
                            old_method_code, new_method_code, old_test_code, rule_reasons
                        )

                        if final_label:
                            records.append({
                                "Project": project,
                                "Commit ID": commit_id,
                                "Label": label,
                                "Needs Update": final_label,
                                "Predict": 1 if "Yes" in final_label else 0,
                                "Reason": reason
                            })

                        # Optional: Sleep to avoid API rate limits
                        time.sleep(1)

                    except FileNotFoundError:
                        continue

df = pd.DataFrame(records)
df.to_excel("hybrid_test_update_detection_results.xlsx", index=False)
print("Results saved to 'hybrid_test_update_detection_results.xlsx'")
