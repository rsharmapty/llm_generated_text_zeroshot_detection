# main.py

import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from datetime import datetime

from src.detector import ZeroShotDetector
from src.utils import load_mage_data_from_hub, generate_text_with_llm

# --- Configuration (Moved to Top) ---
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
PERTURBATION_MODEL_NAME = "t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PERTURBATIONS = 10
TOP_K_TOKENS = 50

HF_TOKEN = "Your hugging face Token here" # Your token here

# --- MAGE Dataset Configuration ---
MAGE_DATASET_NAME = "yaful/MAGE"
MAGE_SPLIT = "test"

# --- Sampling and Length Filters ---
MAX_TEXT_LENGTH_WORDS = 500
NUM_SAMPLES_TO_TAKE_PER_CLASS = 10 # Adjust as needed for faster runs or more data
NUM_GENERATED_SAMPLES = NUM_SAMPLES_TO_TAKE_PER_CLASS

# --- Output Paths (Moved to Top) ---
OUTPUT_DIR = "results"
EXCEL_FILENAME_PREFIX = "zero_shot_detection_results"

# Ensure output directory exists (can be here or inside main())
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metrics(labels, scores, experiment_name, output_dir):
    """
    Plots ROC curve, Precision-Recall curve, and saves the plots with experiment name.
    """
    # ... (function body remains the same) ...
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {experiment_name}')
    plt.legend(loc="lower right")
    plt.grid(True)

    precision, recall, _ = precision_recall_curve(labels, scores)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {experiment_name}')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"detection_curves_{experiment_name.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path)
    print(f"Saved detection curves plot to {plot_path}")
    plt.close()


def plot_dataset_characteristics(dataset, output_dir=OUTPUT_DIR): # OUTPUT_DIR is now defined globally
    """
    Plots characteristics of the loaded dataset.
    """
    # ... (function body remains the same) ...
    df = dataset.to_pandas()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    sns.countplot(x='label', data=df)
    plt.title('Label Distribution (0: Human, 1: LLM)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Human', 'LLM'])

    plt.subplot(1, 3, 2)
    sns.countplot(y='source_model', data=df, order=df['source_model'].value_counts().index)
    plt.title('Source Model Distribution')
    plt.xlabel('Count')
    plt.ylabel('Source Model')

    plt.subplot(1, 3, 3)
    sns.countplot(y='testbed_keyword', data=df, order=df['testbed_keyword'].value_counts().index)
    plt.title('Testbed Keyword Distribution')
    plt.xlabel('Count')
    plt.ylabel('Testbed Keyword')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dataset_characteristics.png")
    plt.savefig(plot_path)
    print(f"Saved dataset characteristics plot to {plot_path}")
    plt.close()


def run_experiment(experiment_name, human_texts, ai_texts, ai_text_sources, detector):
    """
    Runs a single detection experiment and reports metrics.
    """
    # ... (function body remains the same) ...
    print(f"\n--- Running Experiment: {experiment_name} ---")

    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    if not texts:
        print(f"No texts to process for experiment '{experiment_name}'. Skipping evaluation.")
        return pd.DataFrame(), pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC'],
            'Value': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })

    scores = detector.detect(texts, n_perturbations=N_PERTURBATIONS, top_k=TOP_K_TOKENS)
    print("Detection complete.")

    print("\n--- Evaluation Metrics ---")
    predictions = (np.array(scores) > 0).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {roc_auc:.4f}")

    plot_metrics(labels, scores, experiment_name, OUTPUT_DIR)

    detailed_df = pd.DataFrame({
        's.no.': range(1, len(texts) + 1),
        'text_tobe_detected': texts,
        'source_type': ['human'] * len(human_texts) + ai_text_sources,
        'ground_truth': labels,
        'zero_shot_detected_result': predictions,
        'detection_score': scores
    })

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    
    return detailed_df, metrics_df


def main():
    print("--- Starting Zero-Shot Detection Experiment ---")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
        # --- NEW DEBUGGING LINE ---
        total_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"PyTorch reports GPU 0 total memory: {total_gpu_memory_gb:.2f} GB")
        # --- END NEW DEBUGGING LINE ---

    if HF_TOKEN and HF_TOKEN.startswith("hf_"):
        os.environ["HF_TOKEN"] = HF_TOKEN
        print(f"Hugging Face token set as environment variable: {'[TOKEN FOUND]' if os.environ.get('HF_TOKEN') else '[NOT FOUND]'}")
    else:
        print("Warning: HF_TOKEN in main.py is not set or invalid. Model loading may fail.")
        print(f"Current HF_TOKEN in main.py: '{HF_TOKEN}'")

    all_results = {}

    # --- Data Preparation ---
    human_dataset = load_mage_data_from_hub(
        dataset_name=MAGE_DATASET_NAME,
        split=MAGE_SPLIT,
        selected_testbeds_keywords=None,
        selected_source_models=['human'],
        max_text_length_words=MAX_TEXT_LENGTH_WORDS,
        num_samples_to_take=NUM_SAMPLES_TO_TAKE_PER_CLASS
    )
    human_texts = human_dataset['text']
    print(f"Loaded {len(human_texts)} human texts.")

    print("\n--- Generating LLama 3 texts for White-box experiment ---")
    
    if len(human_texts) == 0:
        print("No human texts loaded to use as prompts. Skipping Llama 3 text generation.")
        llama3_generated_texts = []
        llama3_generated_sources = []
    else:
        generation_prompts = human_texts[:NUM_GENERATED_SAMPLES]
        llama3_generated_texts = generate_text_with_llm(
            model_name=LLAMA_MODEL_NAME, 
            device=DEVICE, 
            hf_token=HF_TOKEN, 
            prompts=generation_prompts,
            max_new_tokens=MAX_TEXT_LENGTH_WORDS // 2 
        )
        llama3_generated_texts = [text for text in llama3_generated_texts if text.strip() and len(text.split()) <= MAX_TEXT_LENGTH_WORDS]
        llama3_generated_sources = [f'llama3_{LLAMA_MODEL_NAME.split("/")[-1].replace("-", "_").lower()}'] * len(llama3_generated_texts)
        print(f"Generated {len(llama3_generated_texts)} Llama 3 texts.")

    other_llm_dataset = load_mage_data_from_hub(
        dataset_name=MAGE_DATASET_NAME,
        split=MAGE_SPLIT,
        selected_testbeds_keywords=['test_ood_set_gpt', 'test_ood_set_gpt_para', 'cmv', 'roct', 'eli5', 'writingprompts', 'news'],
        selected_source_models=['gpt-4', 'gpt-3.5-turbo', 'chatgpt', 'flan_t5', 'llama', 'gpt-2', 'generic_machine_llm', 'unclassified_llm'],
        max_text_length_words=MAX_TEXT_LENGTH_WORDS,
        num_samples_to_take=NUM_SAMPLES_TO_TAKE_PER_CLASS,
    )
    other_llm_texts = other_llm_dataset['text']
    other_llm_sources = other_llm_dataset['original_src']
    print(f"Loaded {len(other_llm_texts)} other LLM texts.")

    if len(human_texts) == 0 or (len(llama3_generated_texts) == 0 and len(other_llm_texts) == 0):
        print("Insufficient data loaded or generated for experiments. Exiting.")
        sys.exit(1)

    detector = ZeroShotDetector(model_name=LLAMA_MODEL_NAME, device=DEVICE, perturbation_model_name=PERTURBATION_MODEL_NAME)

    detailed_df_1, metrics_df_1 = run_experiment(
        experiment_name="White-box (Detect Llama 3 by Llama 3)",
        human_texts=human_texts,
        ai_texts=llama3_generated_texts,
        ai_text_sources=llama3_generated_sources,
        detector=detector
    )
    all_results["White-box Results"] = detailed_df_1
    all_results["White-box Metrics"] = metrics_df_1

    detailed_df_2, metrics_df_2 = run_experiment(
        experiment_name="Black-box (Detect Other LLM by Llama 3)",
        human_texts=human_texts,
        ai_texts=other_llm_texts,
        ai_text_sources=other_llm_sources,
        detector=detector
    )
    all_results["Black-box Results"] = detailed_df_2
    all_results["Black-box Metrics"] = metrics_df_2

    print("\n--- Saving all results to Excel ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(OUTPUT_DIR, f"{EXCEL_FILENAME_PREFIX}_full_experiment_{timestamp}.xlsx")

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        for sheet_name, df_to_write in all_results.items():
            df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved '{sheet_name}' to Excel.")

    print(f"All results saved to {excel_path}")
    print("--- Zero-Shot Detection Experiment Complete ---")


if __name__ == "__main__":
    main()
