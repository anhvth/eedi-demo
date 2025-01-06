import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import peft
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from tqdm import tqdm
from contextlib import asynccontextmanager

# ------------------------------
# Utility functions (adapted from your 'run embed code' snippet)
# ------------------------------

MAX_LENGTH = 320


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pools the last token from the hidden states based on the attention mask.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def load_model_and_tokenizer(base_model_path, lora_path=None, load_in_4bit=True):
    """
    Loads a base model and tokenizer. Optionally loads a LoRA adapter.
    """
    # Adjust device_map if you have multiple GPUs or a special inference setup
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map=0,  # Change this based on your actual device setup
        torch_dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_path
    )
    model.resize_token_embeddings(len(tokenizer))
    if lora_path and os.path.exists(lora_path):
        model = peft.PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer


def get_embeddings_in_batches(model, tokenizer, texts, max_length, batch_size=32):
    """
    Embeds a list of texts in batches using the provided model and tokenizer.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


# ------------------------------
# FastAPI application
# ------------------------------

app = FastAPI(
    title="Embedding Service Only",
    version="1.0.0",
    description="A minimal FastAPI app that provides embeddings using a base model (optionally with LoRA).",
)

# CORS middleware (optional, remove or adjust for your use-case)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Pydantic models for request/response
# ------------------------------


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to embed")


class EmbeddingsResponse(BaseModel):
    # Each embedding is a list of floats.
    # For large embeddings, consider using a more compact representation.
    embeddings: List[List[float]]


# ------------------------------
# Global variables: Model & Tokenizer
# ------------------------------

model = None
tokenizer = None

# ------------------------------
# FastAPI Endpoints
# ------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    base_model_path = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-7B")
    lora_model_path = os.getenv("LORA_PATH", None)
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

    print(
        f"Loading model: {base_model_path}, LoRA: {lora_model_path}, 4bit: {load_in_4bit}"
    )
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=base_model_path,
        lora_path=lora_model_path,
        load_in_4bit=load_in_4bit,
    )
    model.to("cuda")
    model.eval()
    yield
    # Cleanup if necessary
    model = None
    tokenizer = None


app = FastAPI(lifespan=lifespan)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest):
    """
    Generate normalized embeddings for the given list of texts.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded.")
    try:
        # Get embeddings
        emb = get_embeddings_in_batches(
            model=model,
            tokenizer=tokenizer,
            texts=request.texts,
            max_length=MAX_LENGTH,
            batch_size=4,  # Adjust based on your resources
        )
        # Convert to Python lists
        emb_list = emb.tolist()
        return EmbeddingsResponse(embeddings=emb_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health")
async def health():
    """
    Health check endpoint.
    """
    return {"message": "Embedding service is running."}


# ------------------------------
# Main entry point
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Override environment variables with command-line arguments if provided
    os.environ["BASE_MODEL_PATH"] = args.base_model
    if args.lora_path:
        os.environ["LORA_PATH"] = args.lora_path
    os.environ["LOAD_IN_4BIT"] = "true" if args.load_in_4bit else "false"

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )
