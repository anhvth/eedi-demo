import argparse
import os

import fire
import peft
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def quantize_model(model_path, output_dir):
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f'Model is quantized and saved at "{output_dir}"')


def save_model(base_model_path, lora_path, merged_output_path):
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_path
    )
    assert len(tokenizer("<response>")["input_ids"]) == 1, (
        lora_path if lora_path else base_model_path
    )
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    # resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    model = peft.PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    if lora_path.endswith("/"):
        lora_path = lora_path[:-1]
    print("Saving auto model")
    model.save_pretrained(merged_output_path)

    model_causal = AutoModelForCausalLM.from_pretrained(
        merged_output_path, torch_dtype=torch.float16
    )
    os.system("rm -r {}".format(merged_output_path))
    print("Saving causal model")

    model_causal.save_pretrained(merged_output_path)
    tokenizer.save_pretrained(merged_output_path)
    print(merged_output_path)


def main(model_path, output_dir, lora_path=None):
    if lora_path:
        merged_output_path = os.path.join(output_dir, "merged")
        awq_output_dir = os.path.join(output_dir, "awq")
        if not os.path.exists(merged_output_path):
            print(f"Merging model from {model_path} and {lora_path}")
            save_model(model_path, lora_path, merged_output_path)
        else:
            print(f"Model already merged at {merged_output_path}")
        model_path = merged_output_path
    else:
        awq_output_dir = output_dir
    print(f"Quantized model will be saved at {awq_output_dir}")

    # if not os.path.exists(awq_output_dir):
    quantize_model(model_path, awq_output_dir)


if __name__ == "__main__":
    fire.Fire(main)
