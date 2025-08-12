# src/utils.py

from datasets import load_dataset, DatasetDict, Dataset
import re
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd

def count_words(text):
    """Counts words in a text by splitting on whitespace."""
    return len(text.split())

def load_mage_data_from_hub(dataset_name, split, selected_testbeds_keywords, max_text_length_words=None, num_samples_to_take=None,
                            selected_source_models=None):
    """
    Loads data from the Hugging Face MAGE dataset, filters by keywords,
    max text length, and optionally by source model.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., "yaful/MAGE").
        split (str): Dataset split (e.g., "test", "validation").
        selected_testbeds_keywords (list): Keywords to filter 'testbed_keyword' column (e.g., ['test_ood_set_gpt', 'human']).
        max_text_length_words (int, optional): Maximum number of words in a text. Texts longer than this are excluded.
        num_samples_to_take (int, optional): Number of samples to take per label (human/LLM).
        selected_source_models (list, optional): List of specific source models to include (e.g., ['gpt2', 'llama2']).

    Returns:
        datasets.Dataset: Filtered dataset.
    """
    print(f"Loading dataset '{dataset_name}' split '{split}'...")
    dataset = load_dataset(dataset_name, split=split)
    
    df = dataset.to_pandas()

    if 'src' not in df.columns:
        print("Error: 'src' column not found in the dataset. Cannot proceed with source parsing.")
        return Dataset.from_pandas(pd.DataFrame({"text": [], "label": [], "source_model": [], "testbed_keyword": [], "original_src": []}))

    df['original_src'] = df['src'].copy()

    df['source_model'] = 'unknown_model'
    df['testbed_keyword'] = 'unknown_testbed'

    debug_parse_counter = 0

    for index, row in df.iterrows():
        src_value = str(row['original_src']) 
        src_value_lower = src_value.lower()
        
        if debug_parse_counter < 20: 
            print(f"DEBUG: Original src: '{src_value}'")

        # Determine source_model
        current_source_model = 'unknown_model'
        if 'human' in src_value_lower:
            current_source_model = 'human'
        elif 'gpt-4' in src_value_lower:
            current_source_model = 'gpt-4'
        elif 'gpt-3.5-turbo' in src_value_lower or 'gpt-3.5-trubo' in src_value_lower:
            current_source_model = 'gpt-3.5-turbo'
        elif 'chatgpt' in src_value_lower:
            current_source_model = 'chatgpt'
        elif 'flan_t5' in src_value_lower:
            current_source_model = 'flan_t5'
        elif 'llama' in src_value_lower:
            current_source_model = 'llama'
        elif 'gpt-2' in src_value_lower:
            current_source_model = 'gpt-2'
        elif 'machine' in src_value_lower:
            current_source_model = 'generic_machine_llm'
        else:
            if row['label'] == 0: # MAGE original label: 0=machine, 1=human
                current_source_model = 'unclassified_llm'
            elif row['label'] == 1: # MAGE original label: 0=machine, 1=human
                current_source_model = 'unclassified_human'
        df.at[index, 'source_model'] = current_source_model

        # Determine testbed_keyword
        current_testbed_keyword = 'unknown_testbed'
        if 'test_ood_set_gpt_para' in src_value_lower:
            current_testbed_keyword = 'test_ood_set_gpt_para'
        elif 'test_ood_set_gpt' in src_value_lower:
            current_testbed_keyword = 'test_ood_set_gpt'
        elif 'cmv' in src_value_lower:
            current_testbed_keyword = 'cmv'
        elif 'roct' in src_value_lower:
            current_testbed_keyword = 'roct'
        elif 'eli5' in src_value_lower:
            current_testbed_keyword = 'eli5'
        elif 'writingprompts' in src_value_lower:
            current_testbed_keyword = 'writingprompts'
        elif 'news' in src_value_lower:
            current_testbed_keyword = 'news'
        elif 'human' in src_value_lower:
            parts = src_value.split('_')
            if len(parts) > 1 and parts[-1].lower() == 'human':
                current_testbed_keyword = "_".join(parts[:-1]).replace('-', '_') 
            else:
                current_testbed_keyword = 'general_human' 
        elif 'machine' in src_value_lower:
            current_testbed_keyword = 'general_machine'
        else:
            current_testbed_keyword = 'unclassified_testbed'
        df.at[index, 'testbed_keyword'] = current_testbed_keyword

        if debug_parse_counter < 20:
            print(f"DEBUG: Parsed source_model: '{df.at[index, 'source_model']}', testbed_keyword: '{df.at[index, 'testbed_keyword']}'")
            debug_parse_counter += 1

    print("\nDEBUG: Initial source_model value_counts (before filters):")
    print(df['source_model'].value_counts())
    print("\nDEBUG: Initial testbed_keyword value_counts (before filters):")
    print(df['testbed_keyword'].value_counts())

    df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)

    filtered_df = df.copy()

    if selected_source_models:
        initial_len = len(filtered_df)
        filtered_df = filtered_df[filtered_df['source_model'].isin(selected_source_models)].copy()
        print(f"Filtered by source_model: {len(filtered_df)} samples remaining from {initial_len}.")
    
    if selected_testbeds_keywords:
        initial_len = len(filtered_df)
        filtered_df = filtered_df[filtered_df['testbed_keyword'].isin(selected_testbeds_keywords)].copy()
        print(f"Filtered by testbed_keyword: {len(filtered_df)} samples remaining from {initial_len}.")

    filtered_df['word_count'] = filtered_df['text'].apply(count_words)
    if max_text_length_words is not None:
        initial_len = len(filtered_df)
        filtered_df = filtered_df[filtered_df['word_count'] <= max_text_length_words].copy()
        print(f"Filtered by max_text_length_words ({max_text_length_words}): {len(filtered_df)} samples remaining from {initial_len}.")

    if num_samples_to_take is not None:
        human_samples = filtered_df[filtered_df['label'] == 0]
        llm_samples = filtered_df[filtered_df['label'] == 1]

        sampled_human = human_samples.sample(min(num_samples_to_take, len(human_samples)), random_state=42)
        sampled_llm = llm_samples.sample(min(num_samples_to_take, len(llm_samples)), random_state=42)
        
        filtered_df = pd.concat([sampled_human, sampled_llm]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(sampled_human)} human and {len(sampled_llm)} LLM texts.")
    
    return Dataset.from_pandas(filtered_df)


def generate_text_with_llm(model_name, device, hf_token, prompts, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    Generates text using a specified LLM.
    """
    print(f"Loading generation model: {model_name} on {device} for text generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        device_map={'': device},
        token=hf_token,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generated_texts = []
    print(f"Generating {len(prompts)} texts...")
    for i, prompt in enumerate(tqdm(prompts, desc="Generating texts")):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1} texts.")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Generation model unloaded.")
    return generated_texts
