"""
Local VLM Server using transformers
Uses Qwen2.5-VL-3B model for OCR post-processing
Memory optimized with 8-bit quantization
"""

import os
import sys
import io
import asyncio
import socket
import time
import torch
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import uvicorn

MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-vl-3b-instruct"
HOST = "0.0.0.0"
PORT = 8001
MIN_PIXELS = int(os.getenv("UGOCR_VLM_MIN_PIXELS", str(128 * 28 * 28)))
MAX_PIXELS = int(os.getenv("UGOCR_VLM_MAX_PIXELS", str(832 * 28 * 28)))
MAX_NEW_TOKENS = int(os.getenv("UGOCR_VLM_MAX_NEW_TOKENS", "512"))
MAX_CONCURRENCY = max(1, int(os.getenv("UGOCR_VLM_MAX_CONCURRENCY", "1")))
WARMUP_ENABLED = os.getenv("UGOCR_VLM_WARMUP", "true").strip().lower() in {"1", "true", "yes", "on"}
WARMUP_MAX_TOKENS = max(1, int(os.getenv("UGOCR_VLM_WARMUP_MAX_TOKENS", "8")))
EMPTY_CACHE_AFTER_REQUEST = os.getenv("UGOCR_VLM_EMPTY_CACHE", "false").strip().lower() in {"1", "true", "yes", "on"}

app = FastAPI(title="Local VLM Server")

model = None
processor = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
model_loaded_at: float | None = None
model_warmed_up = False
last_generate_seconds: float | None = None


def is_port_available(host: str, port: int) -> bool:
    bind_host = "0.0.0.0" if host in {"", "*"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[dict]
    usage: dict


def load_model():
    global model, processor, model_loaded_at
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local VLM server.")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # 8-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quantization_config,
        attn_implementation="eager",
        device_map={"": 0},
    )

    model.eval()
    model_loaded_at = time.time()
    print("Model loaded successfully!")
    if WARMUP_ENABLED:
        warmup_model()


def normalize_messages(messages: list[ChatMessage] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for msg in messages:
        role = msg.role if isinstance(msg, ChatMessage) else str(msg.get("role", "user"))
        content = msg.content if isinstance(msg, ChatMessage) else msg.get("content", "")
        if isinstance(content, list):
            content_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        content_parts.append({"type": "image", "image": image_url})
                    elif part.get("type") == "text":
                        content_parts.append({"type": "text", "text": part.get("text", "")})
            normalized.append({"role": role, "content": content_parts})
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def generate_response(messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, dict[str, int]]:
    global last_generate_seconds
    started_at = time.perf_counter()
    inputs = None
    generated_ids = None
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, MAX_NEW_TOKENS),
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        usage = {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "completion_tokens": int(len(generated_ids_trimmed[0])),
            "total_tokens": int(inputs.input_ids.shape[1] + len(generated_ids_trimmed[0])),
        }
        return response_text, usage
    finally:
        last_generate_seconds = time.perf_counter() - started_at
        del inputs
        del generated_ids
        if EMPTY_CACHE_AFTER_REQUEST:
            torch.cuda.empty_cache()


def warmup_model() -> None:
    global model_warmed_up
    if model is None or processor is None:
        return
    print("Warming up VLM generation...")
    try:
        messages = normalize_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Return OK."}],
                }
            ]
        )
        generate_response(messages, WARMUP_MAX_TOKENS)
        model_warmed_up = True
        print(f"VLM warmup completed in {last_generate_seconds:.3f}s")
    except Exception as exc:
        model_warmed_up = False
        print(f"VLM warmup skipped after failure: {exc}")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "loading",
        "model": "qwen2.5-vl-3b-instruct",
        "device": "cuda",
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_concurrency": MAX_CONCURRENCY,
        "warmup_enabled": WARMUP_ENABLED,
        "warmed_up": model_warmed_up,
        "loaded_seconds_ago": None if model_loaded_at is None else round(time.time() - model_loaded_at, 3),
        "last_generate_seconds": None if last_generate_seconds is None else round(last_generate_seconds, 3),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "qwen2.5-vl-3b-instruct",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        messages = normalize_messages(request.messages)
        async with request_semaphore:
            response_text, usage = generate_response(messages, request.max_tokens)

        return ChatResponse(
            id="chatcmpl-local",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage=usage
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"Starting VLM server on {HOST}:{PORT}")
    if not is_port_available(HOST, PORT):
        print(
            f"Port {PORT} is already in use. VLM server is likely already running. "
            f"Check it with: curl.exe http://127.0.0.1:{PORT}/health"
        )
        raise SystemExit(1)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
