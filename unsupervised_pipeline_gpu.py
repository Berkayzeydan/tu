import os
import re
import json
import time
import pickle
from typing import List, Tuple, Dict, Any

import pandas as pd
import unidecode
import openai
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Good defaults for a strong NVIDIA GPU.
# You can override these from the terminal, for example:
# QWEN_BATCH_SIZE=64 python3 run_unsupervised_three_seed_groups.py
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
QWEN_BATCH_SIZE = int(os.getenv("QWEN_BATCH_SIZE", "32"))
QWEN_MAX_LENGTH = int(os.getenv("QWEN_MAX_LENGTH", "256"))
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "12"))


def configure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script expects a GPU server.")

    # Faster matmul on Ampere/Ada/Hopper GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")


def class_based_collapse(text: str) -> str:
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[A-Z]", "A", text)
    text = re.sub(r"[a-z]", "a", text)
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"(A+)", "A", text)
    text = re.sub(r"(a+)", "a", text)
    text = re.sub(r"(0+)", "0", text)
    return text


def extract_schema_with_patterns(examples):
    formatted = "\n".join(
        [f"{i + 1}. '{row}'" for i, row in enumerate(examples)]
    )

    prompt = f"""
You are given example strings from a single domain.

Examples:
{formatted}

Task:
1. Identify semantic components in each string.
2. Represent each string as an ordered pattern of semantic tags.
3. Build a global set of unique semantic tags.

You MUST internally:
- Separate semantic meaning from syntax (order + separators)
- Merge synonymous components into one tag
- Ensure consistency across all examples

Rules:
- Tags must represent meaning-bearing units ONLY
- Do NOT include separators or punctuation as tags
- Use ONE consistent name per concept
- Use PascalCase
- Minimize tag set while covering all distinct meaning-bearing units

Output JSON:
{{
  "tags": ["Tag1", "Tag2", ...],
  "patterns": [
    {{"example": "...", "pattern": "..."}}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        reasoning_effort="medium",
        verbosity="low"
    )

    rates = {
        "gpt-5-mini": {"in": 0.25, "out": 2.0},
    }

    r = rates["gpt-5-mini"]
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    cost = ((in_tokens * r["in"]) + (out_tokens * r["out"])) / 1_000_000

    usage = {
        "cost": cost,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
    }

    return json.loads(response.choices[0].message.content), usage


def build_schema_prompt(tags):
    return f"""
You are given a fixed set of semantic tags:

{tags}

Task:
For each input string:
- Identify semantic components
- Output ONLY the ordered sequence of tags

Rules:
- Use ONLY the provided tags
- Do NOT invent new tags
- Preserve order exactly
- Output tags separated by spaces
"""


def get_best_dtype():
    # bf16 is usually best on newer strong GPUs. fp16 fallback is fine.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_qwen_model():
    configure_gpu()

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_best_dtype()

    # attn_implementation="flash_attention_2" is fastest if installed.
    # If not installed, we fall back automatically.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=dtype,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        print("Loaded model with FlashAttention 2.")
    except Exception as e:
        print(f"FlashAttention 2 unavailable, using default attention. Reason: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=dtype,
            device_map="cuda",
            trust_remote_code=True,
        )

    model.eval()

    print(f"Model dtype: {dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    return tokenizer, model


def clean_slm_output(decoded: str) -> str:
    # Keep only the generated answer after the final Output: marker.
    if "Output:" in decoded:
        decoded = decoded.split("Output:")[-1]

    decoded = decoded.strip()

    # Remove common chat/template leftovers.
    decoded = decoded.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    # Keep first line only. The model should output a short tag sequence.
    decoded = decoded.splitlines()[0].strip() if decoded else ""

    return decoded


def tag_with_slm(strings, tags, tokenizer, model, batch_size: int = QWEN_BATCH_SIZE):
    schema_prompt = build_schema_prompt(tags)

    prompts = [
        f"{schema_prompt}\nInput: {s}\nOutput:"
        for s in strings
    ]

    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for start_idx in range(0, len(prompts), batch_size):
        batch_number = start_idx // batch_size + 1
        print(f"SLM batch {batch_number}/{total_batches}")

        batch_prompts = prompts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=QWEN_MAX_LENGTH,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_batch = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        for decoded in decoded_batch:
            results.append(clean_slm_output(decoded))

        # Helps avoid memory fragmentation during long multi-dataset runs.
        del inputs, outputs
        torch.cuda.empty_cache()

    return results


def unsupervised_pipeline(df: pd.DataFrame, tokenizer, model, seed=42):
    # Avoid modifying original dataframe from caller.
    df = df.copy()

    examples = df["0"].sample(n=min(5, len(df)), random_state=seed)

    start = time.perf_counter()

    schema, usage = extract_schema_with_patterns(examples)

    tags = schema["tags"]
    print(f"Extracted Tags: {tags}")

    tagged = tag_with_slm(df["0"].tolist(), tags, tokenizer, model)

    df["semantic_pattern"] = tagged
    df["syntactic_pattern"] = df["0"].apply(class_based_collapse)

    grouped = df.groupby(["semantic_pattern", "syntactic_pattern"], sort=False)
    clusters = [group.index.tolist() for _, group in grouped]

    end = time.perf_counter()
    lat = end - start

    return clusters, usage, lat


if __name__ == "__main__":
    tokenizer, model = load_qwen_model()
    print("Qwen model loaded successfully.")
