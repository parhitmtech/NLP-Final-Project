import json
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ============== Setup ==============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Classification Part ======
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    dataset_to_intent = {
        "gsm8k": "math_problem",
        "math": "math_problem",
        "hhh_alignment": "ethical_question",
        "strategyqa": "logical_puzzle",
        "truthful_qa": "multiple_choice_factual",
        "triviaqa": "multiple_choice_factual",
        "hotpotqa": "open_domain_reasoning",
        "mmlu": "education_exam_mcq",
    }
    labels = [dataset_to_intent.get(item["source"], "unknown") for item in data]
    return questions, labels

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vectorize_text(texts):
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy(), embedder

def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# ====== LoRA Adapter + Base Model Part ======
adapters = {
    "gsm8k": "/home/pmathu14/Transformer Model/finetuned-llama-gsm8k-lora/",
    "hhh_alignment": "/home/pmathu14/Transformer Model/finetuned-llama-hhh-lora/",
    "triviaqa": "/home/pmathu14/Transformer Model/finetuned-llama-triviaqa-lora/",
    "mmlu": "/home/pmathu14/Transformer Model/finetuned-llama-mmlu-lora/",
    "strategyqa": "/home/pmathu14/Transformer Model/finetuned-llama-strategyqa-lora/",
    "math": "/home/pmathu14/Transformer Model/finetuned-llama-math-lora/",
    "truthfulqa": "/home/pmathu14/Transformer Model/finetuned-llama-truthfulqa-lora/",
    "hotpotqa": "/home/pmathu14/Transformer Model/finetuned-llama-hotpotqa-lora/",
}

intent_to_dataset = {
    "math_problem": "gsm8k",
    "ethical_question": "hhh_alignment",
    "multiple_choice_factual": "truthfulqa",
    "logical_puzzle": "strategyqa",
    "open_domain_reasoning": "hotpotqa",
    "education_exam_mcq": "mmlu",
}

base_model_name = "meta-llama/Llama-3.2-3B"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

loaded_adapters = {}
current_task = None

def load_adapter(task_name):
    global loaded_adapters, current_task

    if current_task == task_name:
        return loaded_adapters[task_name]

    print(f"Switching adapter to {task_name}")
    adapter_path = adapters.get(task_name)
    if adapter_path is None:
        raise ValueError(f"No adapter path found for task: {task_name}")

    if task_name not in loaded_adapters:
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
        loaded_adapters[task_name] = model_with_adapter

    current_task = task_name
    return loaded_adapters[task_name]

# ====== Inference ======
def classify_prompt(clf, embedder, label_encoder, prompt):
    vec = embedder.encode([prompt], convert_to_tensor=True).cpu().numpy()
    pred = clf.predict(vec)
    intent = label_encoder.inverse_transform(pred)[0]
    dataset = intent_to_dataset.get(intent, "truthfulqa")
    return dataset

def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== Helper function for batch prediction ======
def get_prediction(question):
    task = classify_prompt(clf, embedder, label_encoder, question)
    model = load_adapter(task)
    answer = generate_response(model, question)
    return answer

# ====== Main Control Flow ======
def main():
    global clf, embedder, label_encoder

    dataset_path = "data.json"

    texts, labels = load_data(dataset_path)
    y, label_encoder = encode_labels(labels)
    X, embedder = vectorize_text(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = train_model(X_train, y_train)

    # Evaluate classifier
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # ====== Predict on test_data_student.json ======
    print("Running predictions on test_data_student.json...")
    with open("test_data_student.json", "r") as f:
        questions = json.load(f)

    results = []
    for item in questions:
        question = item["question"]
        answer = get_prediction(question)
        results.append({"prediction": answer.strip()})

    with open("your_output.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to your_output.json ✅")

if __name__ == "__main__":
    main()
