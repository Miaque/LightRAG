import asyncio
import logging
import os
import time
from typing import Optional

import nest_asyncio
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from lightrag import LightRAG, QueryParam
from lightrag.llm import (
    oneapi_complete_if_cache,
    oneapi_embedding,
)
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

app = FastAPI(title="LightRAG API", description="API for RAG operations")

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen2.5-7B-Instruct")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
ONE_API_URL = os.environ.get("ONE_API_URL", "http://172.21.43.92:3000/v1")
print(f"ONE_API_URL: {ONE_API_URL}")
ONE_API_KEY = os.environ.get("ONE_API_KEY", "<KEY>")
print(f"ONE_API_KEY: {ONE_API_KEY}")

# mongo


# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
# milvus


# LLM model function


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await oneapi_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=ONE_API_URL,
        api_key=ONE_API_KEY,
        **kwargs,
    )


# Embedding function


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await oneapi_embedding(
        texts,
        model=EMBEDDING_MODEL,
        base_url=ONE_API_URL,
        api_key=ONE_API_KEY,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"{embedding_dim=}")
    return embedding_dim


# Initialize RAG instance
def get_rag(case_code: str = "default") -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        case_code=case_code,
        llm_model_func=llm_model_func,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=asyncio.run(get_embedding_dim()),
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
        embedding_batch_num=32,
        embedding_func_max_async=16,
        kv_storage="MongoKVStorage",
        graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorge",
    )
    return rag


# Data models


class QueryRequest(BaseModel):
    case_code: str
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    case_code: str
    text: str


class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None


# API routes


@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        start_time = time.perf_counter()
        rag = get_rag(case_code=request.case_code)
        end_time = time.perf_counter()
        logger.info(f"耗时 {end_time - start_time} ms")
        result = await rag.aquery(
            request.query,
            param=QueryParam(
                mode=request.mode, only_need_context=request.only_need_context
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: get_rag(case_code=request.case_code).insert(request.text)
        )
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert_file", response_model=Response)
async def insert_file(case_code: str = Form(), file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: get_rag(case_code=case_code).insert(content)
        )

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file.txt"}'

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
