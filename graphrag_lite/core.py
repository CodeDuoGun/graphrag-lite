# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: graphrag-lite: 极简 GraphRAG 实现

~600 行代码理解 GraphRAG 核心原理

核心功能:
1. 文档插入: text -> chunk -> LLM提取实体/关系 -> embedding -> 存储
2. 查询: query -> 向量检索 -> 构建context -> LLM回答

查询模式:
- local: 从实体出发，检索实体 + 邻居关系
- global: 从关系出发，检索关系 + 涉及实体
- mix: 实体 + 关系 + 原始文本块 (推荐)
- naive: 仅文本块检索 (传统RAG baseline)

特性:
- 批量 Embedding 减少 API 调用
- LLM 响应缓存避免重复请求
- 流式输出支持
- NumPy 加速向量检索
"""

import json
import hashlib
import asyncio
import time
from pathlib import Path
from typing import Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from tqdm import tqdm

from .prompts import ENTITY_EXTRACTION_PROMPT, RAG_RESPONSE_PROMPT
from .utils import chunk_text, top_k_similar, count_tokens, truncate_text
try:
    from .neo4j_store import Neo4jStore
except ImportError:
    Neo4jStore = None  # type: ignore

# Embedding API 重试配置
EMB_MAX_RETRIES = 3
EMB_RETRY_DELAY = 1.0  # 秒
EMB_MAX_BATCH_SIZE = 10  # 单次请求最大文本条数 (部分 API 如 DashScope 限制 ≤10)


class GraphRAGLite:
    def __init__(
        self,
        storage_path: str = "./graphrag_storage",
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        enable_cache: bool = True,
        embedding_fn=None,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
    ):
        """
        初始化 GraphRAGLite
        
        Args:
            storage_path: 存储路径（JSON 文件后端，neo4j 启用时仍用于 LLM 缓存）
            api_key: OpenAI API Key
            base_url: OpenAI API Base URL (可选, 支持兼容 API)
            model: LLM 模型名称
            embedding_model: Embedding 模型名称
            enable_cache: 是否启用 LLM 缓存
            embedding_fn: 自定义 embedding 函数，签名: (texts: list[str]) -> list[list[float]]
                          若提供，则忽略 embedding_model / embedding_api_key / embedding_base_url
            embedding_api_key: 单独给 Embedding 使用的 API Key（与 LLM 不同时使用）
            embedding_base_url: 单独给 Embedding 使用的 Base URL（与 LLM 不同时使用）
            neo4j_uri: Neo4j 连接地址，如 bolt://localhost:7687；若提供则启用 Neo4j 后端
            neo4j_user: Neo4j 用户名
            neo4j_password: Neo4j 密码
            neo4j_database: Neo4j 数据库名，默认 neo4j
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.embedding_model = embedding_model
        self.enable_cache = enable_cache
        self.embedding_fn = embedding_fn  # 自定义 embedding 函数
        
        # OpenAI 客户端 (同步 + 异步) — 用于 LLM
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        
        # Embedding 专用客户端（可与 LLM 使用不同的 base_url / api_key）
        if embedding_fn is None:
            emb_kwargs = {"api_key": embedding_api_key or api_key}
            emb_base_url = embedding_base_url or base_url
            if emb_base_url:
                emb_kwargs["base_url"] = emb_base_url
            self._emb_client = OpenAI(**emb_kwargs)
            self._async_emb_client = AsyncOpenAI(**emb_kwargs)
        else:
            self._emb_client = None
            self._async_emb_client = None
        
        # Neo4j 后端（可选）
        self._neo4j = None
        if neo4j_uri and neo4j_user and neo4j_password:
            if Neo4jStore is None:
                raise ImportError("neo4j 驱动未安装，请执行: pip install neo4j")
            self._neo4j = Neo4jStore(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
            )
            logger.info("[GraphRAGLite] 使用 Neo4j 存储后端")
        
        # 图数据存储（内存缓存，Neo4j 模式下同步写入 Neo4j）
        self.chunks: dict[str, dict] = {}      # chunk_id -> {content, doc_id}
        self.entities: dict[str, dict] = {}    # entity_name -> {type, description}
        self.relations: dict[str, dict] = {}   # "src||tgt" -> {keywords, description}
        
        # Embedding 存储
        self.embeddings: dict[str, list[float]] = {}
        
        # LLM 缓存
        self._llm_cache: dict[str, str] = {}
        
        # 加载数据
        self.load()

    # ==================== LLM 调用 ====================
    
    def _call_llm(self, prompt: str, use_cache: bool = True) -> str:
        """调用 LLM (带缓存)"""
        if use_cache and self.enable_cache:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            if cache_key in self._llm_cache:
                return self._llm_cache[cache_key]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content
        
        if use_cache and self.enable_cache:
            self._llm_cache[cache_key] = result
        
        return result

    def _call_llm_stream(self, prompt: str) -> Generator[str, None, None]:
        """流式调用 LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _get_embedding(self, text: str) -> list[float]:
        """获取单条 embedding (带缓存，用于 query)"""
        cache_key = f"query:{hashlib.md5(text.encode()).hexdigest()}"
        if cache_key in self.embeddings:
            return self.embeddings[cache_key]
        
        emb = self._get_embeddings_batch([text])[0]
        self.embeddings[cache_key] = emb
        return emb

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """批量获取 embedding (减少 API 调用，带重试)
        
        若初始化时传入了 embedding_fn，则直接调用自定义函数；
        否则使用 OpenAI 兼容 API。
        """
        if not texts:
            return []
        
        # 使用自定义 embedding 函数
        if self.embedding_fn is not None:
            return self.embedding_fn(texts)
        
        MAX_TOKENS = 8000
        all_embeddings = []
        batch = []
        batch_tokens = 0
        
        def flush():
            nonlocal batch, batch_tokens, all_embeddings
            if not batch:
                return
            
            # 重试逻辑
            for attempt in range(EMB_MAX_RETRIES):
                try:
                    resp = self._emb_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    all_embeddings.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt < EMB_MAX_RETRIES - 1:
                        logger.warning(f"[Embedding] 第 {attempt + 1} 次失败，{EMB_RETRY_DELAY}s 后重试: {e}")
                        time.sleep(EMB_RETRY_DELAY)
                    else:
                        raise
            
            batch = []
            batch_tokens = 0
        
        for text in texts:
            t = count_tokens(text, self.embedding_model)
            if batch_tokens + t > MAX_TOKENS or len(batch) >= EMB_MAX_BATCH_SIZE:
                flush()
            batch.append(text)
            batch_tokens += t
        
        flush()
        return all_embeddings

    # ==================== 文档插入 ====================
    
    def insert(self, text: str, doc_id: str = None) -> dict:
        """
        插入文档
        
        Args:
            text: 文档文本
            doc_id: 文档ID (可选, 默认用 MD5)
        
        Returns:
            {"doc_id": str, "chunks": int, "entities": int, "relations": int}
        """
        if doc_id is None:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # 1. 分块
        chunks = chunk_text(text)
        logger.info(f"[Insert] Doc_id: {doc_id}, Chunks size: {len(chunks)}")
        
        # 2. 批量获取 chunk embeddings
        chunk_texts = [c["content"] for c in chunks]
        chunk_embs = self._get_embeddings_batch(chunk_texts)
        
        # 3. 处理每个 chunk
        all_entities = []
        all_relations = []
        
        for chunk, emb in zip(chunks, chunk_embs):
            chunk_id = f"{doc_id}_chunk_{chunk['index']}"
            
            self.chunks[chunk_id] = {
                "content": chunk["content"],
                "doc_id": doc_id,
            }
            self.embeddings[f"chunk:{chunk_id}"] = emb
            
            # 提取实体和关系
            entities, relations = self._extract_entities_relations(chunk["content"])
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 4. 批量合并实体和关系
        self._merge_entities_batch(all_entities)
        self._merge_relations_batch(all_relations)
        
        # 5. 保存
        self.save()
        
        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
        }

    def _extract_entities_relations(self, text: str) -> tuple[list[dict], list[dict]]:
        """从文本提取实体和关系"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        result = self._call_llm(prompt)
        
        entities = []
        relations = []
        
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # 去掉 "- " 前缀
            if line.startswith("- "):
                line = line[2:]
            
            # 判断是实体还是关系 (看前缀)
            is_entity = line.lower().startswith("entity:")
            is_relation = line.lower().startswith("relation:")
            
            # 去掉 "Entity: " 或 "Relation: " 前缀
            if is_entity:
                line = line[7:].strip()
            elif is_relation:
                line = line[9:].strip()
            
            parts = line.split("||")
            if len(parts) < 4:
                continue
            
            # 根据前缀或首字段判断类型
            tag = parts[0].strip().lower()
            
            # 实体: [type]||name||type||description 或 entity||name||type||description
            if is_entity or tag in ("entity", "实体"):
                # 如果是 "Entity: type||name||type||desc" 格式
                if is_entity and tag not in ("entity", "实体"):
                    # parts[0] 是 type, parts[1] 是 name
                    entities.append({
                        "name": parts[1].strip(),
                        "type": parts[0].strip(),
                        "description": parts[3].strip() if len(parts) > 3 else parts[2].strip(),
                    })
                else:
                    entities.append({
                        "name": parts[1].strip(),
                        "type": parts[2].strip(),
                        "description": parts[3].strip(),
                    })
            # 关系: relation||src||tgt||keywords||description
            elif is_relation or tag in ("relation", "关系"):
                if len(parts) >= 5:
                    relations.append({
                        "src": parts[1].strip(),
                        "tgt": parts[2].strip(),
                        "keywords": parts[3].strip(),
                        "description": parts[4].strip(),
                    })
        
        return entities, relations

    def _merge_entities_batch(self, entities: list[dict]) -> None:
        """批量合并实体 (批量计算 embedding)"""
        if not entities:
            return
        
        MAX_DESC_TOKENS = 2000
        
        # 按名称分组
        grouped = {}
        for e in entities:
            name = e["name"]
            if name not in grouped:
                grouped[name] = {"type": e["type"], "descriptions": []}
            grouped[name]["descriptions"].append(e["description"])
        
        # 准备批量 embedding
        names = []
        texts = []
        
        for name, data in grouped.items():
            if name in self.entities:
                existing_desc = self.entities[name]["description"]
                new_desc = " ".join(data["descriptions"])
                merged_desc = f"{existing_desc} {new_desc}"
            else:
                merged_desc = " ".join(data["descriptions"])
            
            # 截断过长描述
            merged_desc = truncate_text(merged_desc, MAX_DESC_TOKENS, self.embedding_model)
            
            self.entities[name] = {
                "type": data["type"],
                "description": merged_desc,
            }
            
            names.append(name)
            texts.append(f"{name}: {merged_desc}")
        
        # 批量获取 embedding
        embs = self._get_embeddings_batch(texts)
        for name, emb in zip(names, embs):
            self.embeddings[f"entity:{name}"] = emb

    def _merge_relations_batch(self, relations: list[dict]) -> None:
        """批量合并关系"""
        if not relations:
            return
        
        MAX_DESC_TOKENS = 2000
        
        # 按 src||tgt 分组
        grouped = {}
        for r in relations:
            key = f"{r['src']}||{r['tgt']}"
            if key not in grouped:
                grouped[key] = {"keywords": set(), "descriptions": []}
            grouped[key]["keywords"].update(r["keywords"].split())
            grouped[key]["descriptions"].append(r["description"])
        
        # 准备批量 embedding
        keys = []
        texts = []
        
        for key, data in grouped.items():
            if key in self.relations:
                existing = self.relations[key]
                keywords = existing["keywords"] + " " + " ".join(data["keywords"])
                description = existing["description"] + " " + " ".join(data["descriptions"])
            else:
                keywords = " ".join(data["keywords"])
                description = " ".join(data["descriptions"])
            
            # 截断过长描述
            description = truncate_text(description, MAX_DESC_TOKENS, self.embedding_model)
            
            self.relations[key] = {
                "keywords": keywords,
                "description": description,
            }
            
            keys.append(key)
            texts.append(f"{key}: {description}")
        
        # 批量获取 embedding
        embs = self._get_embeddings_batch(texts)
        for key, emb in zip(keys, embs):
            self.embeddings[f"relation:{key}"] = emb

    # ==================== 查询 ====================
    
    def query(
        self, 
        question: str, 
        mode: str = "mix", 
        top_k: int = 10,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        查询
        
        Args:
            question: 用户问题
            mode: "local" / "global" / "mix" / "naive"
            top_k: 检索数量
            stream: 是否流式输出
        
        Returns:
            str 或 Generator (stream=True)
        """
        # 1. 根据模式检索
        if mode == "local":
            context = self.local_search(question, top_k)
        elif mode == "global":
            context = self.global_search(question, top_k)
        elif mode == "mix":
            context = self.mix_search(question, top_k)
        elif mode == "naive":
            context = self.naive_search(question, top_k)
        else:
            raise ValueError(f"不支持的模式: {mode}, 请使用 local / global / mix / naive")
        
        if not context:
            return "未找到相关知识，无法回答。"
        
        # 2. 生成回答
        prompt = RAG_RESPONSE_PROMPT.format(context=context, query=question)
        
        if stream:
            logger.debug(f"prompt: {prompt}")
            return self._call_llm_stream(prompt)
        else:
            return self._call_llm(prompt, use_cache=False)

    def local_search(self, query: str, top_k: int, query_emb: list[float] = None) -> str:
        """Local 搜索: 从实体出发，检索实体 + 邻居关系
        
        Args:
            query: 查询文本
            top_k: 检索数量
            query_emb: 预计算的 query embedding (可选，用于异步场景)
        """
        if not self.entities:
            return ""
        
        if query_emb is None:
            query_emb = self._get_embedding(query)
        
        # 批量检索实体 (NumPy 加速)
        entity_keys = list(self.entities.keys())
        entity_vecs = [self.embeddings.get(f"entity:{k}", []) for k in entity_keys]
        valid_keys = [k for k, v in zip(entity_keys, entity_vecs) if v]
        valid_vecs = [v for v in entity_vecs if v]
        
        top_entities = top_k_similar(query_emb, valid_keys, valid_vecs, top_k)
        
        # 构建上下文 (带编号)
        context_parts = ["=== Entities ==="]
        for idx, (name, score) in enumerate(top_entities):
            data = self.entities[name]
            context_parts.append(f"({idx}) {name} [{data['type']}]: {data['description']}")
        
        # 获取相关关系
        top_entity_names = {e[0] for e in top_entities}
        related_relations = []
        for key in self.relations:
            src, tgt = key.split("||")
            if src in top_entity_names or tgt in top_entity_names:
                related_relations.append(key)
        
        if related_relations:
            context_parts.append("\n=== Relationships ===")
            for idx, key in enumerate(related_relations[:top_k]):
                src, tgt = key.split("||")
                data = self.relations[key]
                context_parts.append(f"({idx}) {src} -> {tgt}: {data['description']}")
        
        return "\n".join(context_parts)

    def global_search(self, query: str, top_k: int, query_emb: list[float] = None) -> str:
        """Global 搜索: 从关系出发，检索关系 + 涉及实体
        
        Args:
            query: 查询文本
            top_k: 检索数量
            query_emb: 预计算的 query embedding (可选，用于异步场景)
        """
        if not self.relations:
            return ""
        
        if query_emb is None:
            query_emb = self._get_embedding(query)
        
        # 批量检索关系
        relation_keys = list(self.relations.keys())
        relation_vecs = [self.embeddings.get(f"relation:{k}", []) for k in relation_keys]
        valid_keys = [k for k, v in zip(relation_keys, relation_vecs) if v]
        valid_vecs = [v for v in relation_vecs if v]
        
        top_relations = top_k_similar(query_emb, valid_keys, valid_vecs, top_k)
        
        # 构建上下文 (带编号)
        context_parts = ["=== Relationships ==="]
        for idx, (key, score) in enumerate(top_relations):
            src, tgt = key.split("||")
            data = self.relations[key]
            context_parts.append(f"({idx}) {src} -> {tgt}: {data['description']}")
        
        # 获取涉及的实体
        involved_entities = set()
        for key, _ in top_relations:
            src, tgt = key.split("||")
            involved_entities.add(src)
            involved_entities.add(tgt)
        
        if involved_entities:
            context_parts.append("\n=== Entities ===")
            for idx, name in enumerate(involved_entities):
                if name in self.entities:
                    data = self.entities[name]
                    context_parts.append(f"({idx}) {name} [{data['type']}]: {data['description']}")
        
        return "\n".join(context_parts)

    def mix_search(self, query: str, top_k: int, query_emb: list[float] = None) -> str:
        """Mix 搜索: 实体 + 关系 + 原始文本块 (推荐)
        
        Args:
            query: 查询文本
            top_k: 检索数量
            query_emb: 预计算的 query embedding (可选，用于异步场景)
        """
        if query_emb is None:
            query_emb = self._get_embedding(query)
        
        third_k = max(1, top_k // 3)
        context_parts = []
        
        # 1. 检索实体
        if self.entities:
            entity_keys = list(self.entities.keys())
            entity_vecs = [self.embeddings.get(f"entity:{k}", []) for k in entity_keys]
            valid_entity_keys = [k for k, v in zip(entity_keys, entity_vecs) if v]
            valid_entity_vecs = [v for v in entity_vecs if v]
            top_entities = top_k_similar(query_emb, valid_entity_keys, valid_entity_vecs, third_k)
            
            if top_entities:
                context_parts.append("=== Entities ===")
                for idx, (name, _) in enumerate(top_entities):
                    if name in self.entities:
                        data = self.entities[name]
                        context_parts.append(f"({idx}) {name} [{data['type']}]: {data['description']}")
        
        # 2. 检索关系
        if self.relations:
            relation_keys = list(self.relations.keys())
            relation_vecs = [self.embeddings.get(f"relation:{k}", []) for k in relation_keys]
            valid_relation_keys = [k for k, v in zip(relation_keys, relation_vecs) if v]
            valid_relation_vecs = [v for v in relation_vecs if v]
            top_relations = top_k_similar(query_emb, valid_relation_keys, valid_relation_vecs, third_k)
            
            if top_relations:
                context_parts.append("\n=== Relationships ===")
                for idx, (key, _) in enumerate(top_relations):
                    if key in self.relations:
                        src, tgt = key.split("||")
                        data = self.relations[key]
                        context_parts.append(f"({idx}) {src} -> {tgt}: {data['description']}")
        
        # 3. 检索原始文本块
        if self.chunks:
            chunk_keys = list(self.chunks.keys())
            chunk_vecs = [self.embeddings.get(f"chunk:{k}", []) for k in chunk_keys]
            valid_chunk_keys = [k for k, v in zip(chunk_keys, chunk_vecs) if v]
            valid_chunk_vecs = [v for v in chunk_vecs if v]
            top_chunks = top_k_similar(query_emb, valid_chunk_keys, valid_chunk_vecs, third_k)
            
            if top_chunks:
                context_parts.append("\n=== Sources ===")
                for idx, (chunk_id, _) in enumerate(top_chunks):
                    if chunk_id in self.chunks:
                        content = self.chunks[chunk_id]["content"]
                        if len(content) > 1000:
                            content = content[:1000] + "..."
                        context_parts.append(f"({idx}) {content}")
        
        return "\n".join(context_parts)

    def naive_search(self, query: str, top_k: int, query_emb: list[float] = None) -> str:
        """Naive 搜索: 仅原始文本块检索 (传统 RAG baseline)
        
        Args:
            query: 查询文本
            top_k: 检索数量
            query_emb: 预计算的 query embedding (可选，用于异步场景)
        """
        if not self.chunks:
            return ""
        
        if query_emb is None:
            query_emb = self._get_embedding(query)
        
        chunk_keys = list(self.chunks.keys())
        chunk_vecs = [self.embeddings.get(f"chunk:{k}", []) for k in chunk_keys]
        valid_chunk_keys = [k for k, v in zip(chunk_keys, chunk_vecs) if v]
        valid_chunk_vecs = [v for v in chunk_vecs if v]
        
        top_chunks = top_k_similar(query_emb, valid_chunk_keys, valid_chunk_vecs, top_k)
        
        context_parts = ["=== Sources ==="]
        for idx, (chunk_id, _) in enumerate(top_chunks):
            if chunk_id in self.chunks:
                content = self.chunks[chunk_id]["content"]
                context_parts.append(f"({idx}) {content}")
        
        return "\n".join(context_parts)

    # ==================== 持久化 ====================
    
    def save(self) -> None:
        """保存数据（JSON 文件 + Neo4j 双写，无 Neo4j 时仅写文件）"""
        # ---- Neo4j 后端 ----
        if self._neo4j is not None:
            self._save_to_neo4j()
        
        # ---- JSON 文件后端（始终保留，用于 LLM 缓存及离线备份）----
        graph_data = {
            "chunks": self.chunks,
            "entities": self.entities,
            "relations": self.relations,
        }
        graph_path = self.storage_path / "graph_data.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        emb_path = self.storage_path / "embeddings.jsonl"
        with open(emb_path, "w", encoding="utf-8") as f:
            for key, emb in self.embeddings.items():
                f.write(json.dumps({"key": key, "embedding": emb}, ensure_ascii=False) + "\n")
        
        if self.enable_cache and self._llm_cache:
            cache_path = self.storage_path / "llm_cache.json"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self._llm_cache, f, ensure_ascii=False)
        
        logger.info(f"[Save] Graph data saved: {self.storage_path.resolve()}")

    def _save_to_neo4j(self) -> None:
        """将内存数据全量同步到 Neo4j"""
        neo = self._neo4j
        # 实体
        for name, data in self.entities.items():
            emb = self.embeddings.get(f"entity:{name}")
            neo.upsert_entity(name, data["type"], data["description"], emb)
        # 关系
        for key, data in self.relations.items():
            src, tgt = key.split("||")
            emb = self.embeddings.get(f"relation:{key}")
            neo.upsert_relation(src, tgt, data["keywords"], data["description"], emb)
        # 文本块
        for chunk_id, data in self.chunks.items():
            emb = self.embeddings.get(f"chunk:{chunk_id}")
            neo.upsert_chunk(chunk_id, data["content"], data["doc_id"], emb)
        # query embedding 缓存
        for key, emb in self.embeddings.items():
            if key.startswith("query:"):
                neo.upsert_embedding(key, emb)
        logger.info("[Neo4j] 数据已同步")

    def load(self) -> None:
        """加载数据（优先从 Neo4j，否则从 JSON 文件）"""
        if self._neo4j is not None:
            self._load_from_neo4j()
        else:
            self._load_from_json()

    def _load_from_neo4j(self) -> None:
        """从 Neo4j 加载数据到内存"""
        neo = self._neo4j
        # 实体
        for e in neo.list_entities():
            self.entities[e["name"]] = {"type": e["type"], "description": e["description"]}
            if e["embedding"]:
                self.embeddings[f"entity:{e['name']}"] = e["embedding"]
        # 关系
        for r in neo.list_relations():
            key = f"{r['src']}||{r['tgt']}"
            self.relations[key] = {"keywords": r["keywords"], "description": r["description"]}
            if r["embedding"]:
                self.embeddings[f"relation:{key}"] = r["embedding"]
        # 文本块
        for c in neo.list_chunks():
            self.chunks[c["chunk_id"]] = {"content": c["content"], "doc_id": c["doc_id"]}
            if c["embedding"]:
                self.embeddings[f"chunk:{c['chunk_id']}"] = c["embedding"]
        # query embedding 缓存
        for key, emb in neo.list_embeddings().items():
            self.embeddings[key] = emb
        logger.info(
            f"[Neo4j Load] {len(self.chunks)} chunks, "
            f"{len(self.entities)} entities, {len(self.relations)} relations"
        )
        # LLM 缓存仍从 JSON 文件读取
        cache_path = self.storage_path / "llm_cache.json"
        if self.enable_cache and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self._llm_cache = json.load(f)

    def _load_from_json(self) -> None:
        """从 JSON 文件加载数据"""
        graph_path = self.storage_path / "graph_data.json"
        emb_path = self.storage_path / "embeddings.jsonl"
        cache_path = self.storage_path / "llm_cache.json"
        
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", {})
            self.entities = data.get("entities", {})
            self.relations = data.get("relations", {})
            logger.info(f"[Load] Graph data: {len(self.chunks)} chunks, {len(self.entities)} entities, {len(self.relations)} relations")
        
        if emb_path.exists():
            self.embeddings = {}
            with open(emb_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.embeddings[item["key"]] = item["embedding"]
            logger.info(f"[Load] Embeddings: {len(self.embeddings)}")
        
        if self.enable_cache and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self._llm_cache = json.load(f)

    # ==================== 辅助方法 ====================
    
    def has_data(self) -> bool:
        """检查是否有数据"""
        return len(self.entities) > 0 or len(self.relations) > 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "embeddings": len(self.embeddings),
            "llm_cache": len(self._llm_cache),
        }
    
    def get_entity(self, name: str) -> dict | None:
        """获取实体"""
        return self.entities.get(name)
    
    def get_relation(self, src: str, tgt: str) -> dict | None:
        """获取关系"""
        return self.relations.get(f"{src}||{tgt}")
    
    def list_entities(self) -> list[str]:
        """列出所有实体"""
        return list(self.entities.keys())
    
    def list_relations(self) -> list[tuple[str, str]]:
        """列出所有关系"""
        return [tuple(k.split("||")) for k in self.relations.keys()]
    
    def clear(self) -> None:
        """清空所有数据（内存 + JSON 文件 + Neo4j）"""
        self.chunks = {}
        self.entities = {}
        self.relations = {}
        self.embeddings = {}
        self._llm_cache = {}
        
        for fname in ["graph_data.json", "embeddings.jsonl", "llm_cache.json"]:
            p = self.storage_path / fname
            if p.exists():
                p.unlink()
        
        if self._neo4j is not None:
            self._neo4j.clear_all()
        
        logger.info("[Clear] 数据已清空")

    # ==================== Neo4j 增删改查 API ====================

    def neo4j_add_entity(self, name: str, entity_type: str, description: str) -> None:
        """
        向 Neo4j 新增实体并更新内存。
        需要先调用 _get_embeddings_batch 获取向量，此处自动计算。
        """
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j，请在初始化时传入 neo4j_uri/user/password")
        emb = self._get_embeddings_batch([f"{name}: {description}"])[0]
        self.entities[name] = {"type": entity_type, "description": description}
        self.embeddings[f"entity:{name}"] = emb
        self._neo4j.upsert_entity(name, entity_type, description, emb)
        logger.info(f"[Neo4j] 新增实体: {name}")

    def neo4j_update_entity(self, name: str, description: str) -> bool:
        """更新 Neo4j 实体描述（同步内存）"""
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j")
        if name not in self.entities:
            return False
        emb = self._get_embeddings_batch([f"{name}: {description}"])[0]
        self.entities[name]["description"] = description
        self.embeddings[f"entity:{name}"] = emb
        return self._neo4j.update_entity(name, description, emb)

    def neo4j_delete_entity(self, name: str) -> bool:
        """删除 Neo4j 实体（同步内存）"""
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j")
        self.entities.pop(name, None)
        self.embeddings.pop(f"entity:{name}", None)
        return self._neo4j.delete_entity(name)

    def neo4j_add_relation(self, src: str, tgt: str, keywords: str, description: str) -> None:
        """向 Neo4j 新增关系（同步内存）"""
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j")
        key = f"{src}||{tgt}"
        emb = self._get_embeddings_batch([f"{key}: {description}"])[0]
        self.relations[key] = {"keywords": keywords, "description": description}
        self.embeddings[f"relation:{key}"] = emb
        self._neo4j.upsert_relation(src, tgt, keywords, description, emb)
        logger.info(f"[Neo4j] 新增关系: {src} -> {tgt}")

    def neo4j_delete_relation(self, src: str, tgt: str) -> bool:
        """删除 Neo4j 关系（同步内存）"""
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j")
        key = f"{src}||{tgt}"
        self.relations.pop(key, None)
        self.embeddings.pop(f"relation:{key}", None)
        return self._neo4j.delete_relation(src, tgt)

    def neo4j_delete_doc(self, doc_id: str) -> int:
        """按文档 ID 删除所有 Chunk（同步内存），返回删除数量"""
        if self._neo4j is None:
            raise RuntimeError("未配置 Neo4j")
        to_del = [cid for cid, v in self.chunks.items() if v["doc_id"] == doc_id]
        for cid in to_del:
            self.chunks.pop(cid)
            self.embeddings.pop(f"chunk:{cid}", None)
        count = self._neo4j.delete_chunks_by_doc(doc_id)
        logger.info(f"[Neo4j] 删除文档 {doc_id}，共 {count} 个 Chunk")
        return count

    # ==================== 异步方法 ====================
    
    async def _acall_llm(self, prompt: str, use_cache: bool = True) -> str:
        """异步调用 LLM (带缓存)"""
        if use_cache and self.enable_cache:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            if cache_key in self._llm_cache:
                logger.debug("[Cache Hit] LLM 响应")
                return self._llm_cache[cache_key]
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content
        
        if use_cache and self.enable_cache:
            self._llm_cache[cache_key] = result
        
        return result

    async def _acall_llm_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """异步流式调用 LLM"""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _aget_embedding(self, text: str) -> list[float]:
        """异步获取单条 embedding (带缓存，用于 query)"""
        cache_key = f"query:{hashlib.md5(text.encode()).hexdigest()}"
        if cache_key in self.embeddings:
            return self.embeddings[cache_key]
        
        emb = (await self._aget_embeddings_batch([text]))[0]
        self.embeddings[cache_key] = emb
        return emb

    async def _aget_embeddings_batch(self, texts: list[str], show_progress: bool = False, desc: str = "Embeddings") -> list[list[float]]:
        """异步批量获取 embedding (带重试)
        
        若初始化时传入了 embedding_fn，则在线程池中调用自定义函数；
        否则使用 OpenAI 兼容 API。
        """
        if not texts:
            return []
        
        # 使用自定义 embedding 函数（在线程池中执行，避免阻塞事件循环）
        if self.embedding_fn is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embedding_fn, texts)
        
        MAX_TOKENS = 8000
        all_embeddings = []
        batch = []
        batch_tokens = 0
        pbar = tqdm(total=len(texts), desc=desc, disable=not show_progress)
        
        async def flush():
            nonlocal batch, batch_tokens, all_embeddings
            if not batch:
                return
            
            # 重试逻辑
            for attempt in range(EMB_MAX_RETRIES):
                try:
                    resp = await self._async_emb_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    all_embeddings.extend([d.embedding for d in resp.data])
                    pbar.update(len(batch))
                    break
                except Exception as e:
                    if attempt < EMB_MAX_RETRIES - 1:
                        logger.warning(f"[Embedding] 第 {attempt + 1} 次失败，{EMB_RETRY_DELAY}s 后重试: {e}")
                        await asyncio.sleep(EMB_RETRY_DELAY)
                    else:
                        raise
            
            batch = []
            batch_tokens = 0
        
        for text in texts:
            t = count_tokens(text, self.embedding_model)
            if batch_tokens + t > MAX_TOKENS or len(batch) >= EMB_MAX_BATCH_SIZE:
                await flush()
            batch.append(text)
            batch_tokens += t
        
        await flush()
        pbar.close()
        return all_embeddings

    async def _aextract_entities_relations(self, text: str) -> tuple[list[dict], list[dict]]:
        """异步从文本提取实体和关系"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        result = await self._acall_llm(prompt)
        
        entities = []
        relations = []
        
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("- "):
                line = line[2:]
            
            is_entity = line.lower().startswith("entity:")
            is_relation = line.lower().startswith("relation:")
            
            if is_entity:
                line = line[7:].strip()
            elif is_relation:
                line = line[9:].strip()
            
            parts = line.split("||")
            if len(parts) < 4:
                continue
            
            tag = parts[0].strip().lower()
            
            if is_entity or tag in ("entity", "实体"):
                if is_entity and tag not in ("entity", "实体"):
                    entities.append({
                        "name": parts[1].strip(),
                        "type": parts[0].strip(),
                        "description": parts[3].strip() if len(parts) > 3 else parts[2].strip(),
                    })
                else:
                    entities.append({
                        "name": parts[1].strip(),
                        "type": parts[2].strip(),
                        "description": parts[3].strip(),
                    })
            elif is_relation or tag in ("relation", "关系"):
                if len(parts) >= 5:
                    relations.append({
                        "src": parts[1].strip(),
                        "tgt": parts[2].strip(),
                        "keywords": parts[3].strip(),
                        "description": parts[4].strip(),
                    })
        
        return entities, relations

    async def _amerge_entities_batch(self, entities: list[dict], show_progress: bool = False) -> None:
        """异步批量合并实体"""
        if not entities:
            return
        
        MAX_DESC_TOKENS = 2000
        
        grouped = {}
        for e in entities:
            name = e["name"]
            if name not in grouped:
                grouped[name] = {"type": e["type"], "descriptions": []}
            grouped[name]["descriptions"].append(e["description"])
        
        names = []
        texts = []
        
        for name, data in grouped.items():
            if name in self.entities:
                existing_desc = self.entities[name]["description"]
                new_desc = " ".join(data["descriptions"])
                merged_desc = f"{existing_desc} {new_desc}"
            else:
                merged_desc = " ".join(data["descriptions"])
            
            # 截断过长描述
            merged_desc = truncate_text(merged_desc, MAX_DESC_TOKENS, self.embedding_model)
            
            self.entities[name] = {
                "type": data["type"],
                "description": merged_desc,
            }
            
            names.append(name)
            texts.append(f"{name}: {merged_desc}")
        
        embs = await self._aget_embeddings_batch(texts, show_progress=show_progress, desc="Entity Embeddings")
        for name, emb in zip(names, embs):
            self.embeddings[f"entity:{name}"] = emb

    async def _amerge_relations_batch(self, relations: list[dict], show_progress: bool = False) -> None:
        """异步批量合并关系"""
        if not relations:
            return
        
        MAX_DESC_TOKENS = 2000
        
        grouped = {}
        for r in relations:
            key = f"{r['src']}||{r['tgt']}"
            if key not in grouped:
                grouped[key] = {"keywords": set(), "descriptions": []}
            grouped[key]["keywords"].update(r["keywords"].split())
            grouped[key]["descriptions"].append(r["description"])
        
        keys = []
        texts = []
        
        for key, data in grouped.items():
            if key in self.relations:
                existing = self.relations[key]
                keywords = existing["keywords"] + " " + " ".join(data["keywords"])
                description = existing["description"] + " " + " ".join(data["descriptions"])
            else:
                keywords = " ".join(data["keywords"])
                description = " ".join(data["descriptions"])
            
            # 截断过长描述
            description = truncate_text(description, MAX_DESC_TOKENS, self.embedding_model)
            
            self.relations[key] = {
                "keywords": keywords,
                "description": description,
            }
            
            keys.append(key)
            texts.append(f"{key}: {description}")
        
        embs = await self._aget_embeddings_batch(texts, show_progress=show_progress, desc="Relation Embeddings")
        for key, emb in zip(keys, embs):
            self.embeddings[f"relation:{key}"] = emb

    async def ainsert(self, text: str, doc_id: str = None, show_progress: bool = True) -> dict:
        """
        异步插入文档
        
        Args:
            text: 文档文本
            doc_id: 文档ID (可选, 默认用 MD5)
            show_progress: 是否显示进度条
        
        Returns:
            {"doc_id": str, "chunks": int, "entities": int, "relations": int}
        """
        if doc_id is None:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # 1. 分块
        chunks = chunk_text(text)
        logger.info(f"[Async Insert] 文档 {doc_id} 分块: {len(chunks)} 块")
        
        # 2. 异步批量获取 chunk embeddings (带进度条)
        chunk_texts = [c["content"] for c in chunks]
        if show_progress:
            logger.info("[Step 1/4] 获取文本 Embeddings...")
        chunk_embs = await self._aget_embeddings_batch(chunk_texts, show_progress=show_progress, desc="Chunk Embeddings")
        
        # 3. 并发处理每个 chunk 的实体关系提取
        all_entities = []
        all_relations = []
        
        if show_progress:
            logger.info("[Step 2/4] 提取实体和关系...")
            pbar = tqdm(total=len(chunks), desc="Extracting Entities")
        
        async def process_chunk(chunk, emb, idx):
            chunk_id = f"{doc_id}_chunk_{chunk['index']}"
            self.chunks[chunk_id] = {
                "content": chunk["content"],
                "doc_id": doc_id,
            }
            self.embeddings[f"chunk:{chunk_id}"] = emb
            result = await self._aextract_entities_relations(chunk["content"])
            if show_progress:
                pbar.update(1)
            return result
        
        # 并发提取实体关系
        tasks = [process_chunk(chunk, emb, idx) for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embs))]
        results = await asyncio.gather(*tasks)
        
        if show_progress:
            pbar.close()
        
        for entities, relations in results:
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 4. 异步批量合并实体和关系
        if show_progress:
            logger.info(f"[Step 3/4] 合并实体 ({len(all_entities)} 个)...")
        await self._amerge_entities_batch(all_entities, show_progress=show_progress)
        
        if show_progress:
            logger.info(f"[Step 4/4] 合并关系 ({len(all_relations)} 个)...")
        await self._amerge_relations_batch(all_relations, show_progress=show_progress)
        
        # 5. 保存
        self.save()
        
        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
        }

    async def aquery(
        self, 
        question: str, 
        mode: str = "mix", 
        top_k: int = 10,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        异步查询
        
        Args:
            question: 用户问题
            mode: "local" / "global" / "mix" / "naive"
            top_k: 检索数量
            stream: 是否流式输出
        
        Returns:
            str 或 AsyncGenerator (stream=True)
        """
        # 1. 异步获取 query embedding (带缓存)
        query_emb = await self._aget_embedding(question)
        
        # 2. 根据模式检索 (复用同步方法，传入预计算的 embedding)
        if mode == "local":
            context = self.local_search(question, top_k, query_emb=query_emb)
        elif mode == "global":
            context = self.global_search(question, top_k, query_emb=query_emb)
        elif mode == "mix":
            context = self.mix_search(question, top_k, query_emb=query_emb)
        elif mode == "naive":
            context = self.naive_search(question, top_k, query_emb=query_emb)
        else:
            raise ValueError(f"不支持的模式: {mode}, 请使用 local / global / mix / naive")
        
        if not context:
            return "未找到相关知识，无法回答。"
        
        # 3. 生成回答
        prompt = RAG_RESPONSE_PROMPT.format(context=context, query=question)
        
        if stream:
            return self._acall_llm_stream(prompt)
        else:
            return await self._acall_llm(prompt, use_cache=False)
