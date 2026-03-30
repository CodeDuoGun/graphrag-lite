# -*- coding: utf-8 -*-
"""
@description: Neo4j 存储后端

图数据库模型:
  - Node (:Entity)   {name, type, description, embedding}
  - Node (:Chunk)    {chunk_id, content, doc_id, embedding}
  - Rel  [:RELATION] {keywords, description, embedding}
  - Node (:EmbCache) {key, embedding}  # query embedding 缓存

增删改查 API:
  upsert_entity / delete_entity / get_entity / list_entities
  upsert_relation / delete_relation / get_relation / list_relations
  upsert_chunk / delete_chunk / delete_chunks_by_doc / get_chunk / list_chunks
  upsert_embedding / get_embedding / list_embeddings
  get_stats / clear_all
"""

from __future__ import annotations
import json
from typing import Optional
from loguru import logger

try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    raise ImportError("neo4j 驱动未安装，请执行: pip install neo4j")


class Neo4jStore:
    """
    Neo4j 存储后端，封装 GraphRAGLite 所需的全部增删改查操作。

    示例::

        store = Neo4jStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password",
        )
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_indexes()
        logger.info(f"[Neo4j] 已连接: {uri}, 数据库: {database}")

    def close(self):
        """关闭连接"""
        self._driver.close()

    # -------------------- 内部工具 --------------------

    def _run(self, cypher: str, **params):
        """执行 Cypher，返回 records 列表"""
        with self._driver.session(database=self.database) as session:
            return session.run(cypher, **params).data()

    def _ensure_indexes(self):
        """建立唯一约束和索引"""
        stmts = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT embcache_key IF NOT EXISTS FOR (e:EmbCache) REQUIRE e.key IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        for stmt in stmts:
            try:
                self._run(stmt)
            except Exception as ex:
                logger.debug(f"[Neo4j] index/constraint: {ex}")

    # -------------------- 实体 (Entity) --------------------

    def upsert_entity(self, name: str, entity_type: str, description: str,
                      embedding: list[float] = None) -> None:
        """插入或更新实体节点"""
        self._run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $entity_type,
                e.description = $description,
                e.embedding = $embedding
            """,
            name=name, entity_type=entity_type, description=description,
            embedding=json.dumps(embedding) if embedding is not None else None,
        )

    def delete_entity(self, name: str) -> bool:
        """删除实体及其所有关联关系。返回是否实际删除"""
        result = self._run(
            "MATCH (e:Entity {name: $name}) DETACH DELETE e RETURN count(e) AS n",
            name=name,
        )
        return (result[0]["n"] if result else 0) > 0

    def get_entity(self, name: str) -> Optional[dict]:
        """获取单个实体，返回 {name, type, description, embedding} 或 None"""
        result = self._run(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.name AS name, e.type AS type, e.description AS description, e.embedding AS embedding
            """,
            name=name,
        )
        return self._parse_entity_row(result[0]) if result else None

    def list_entities(self) -> list[dict]:
        """列出所有实体"""
        result = self._run(
            """
            MATCH (e:Entity)
            RETURN e.name AS name, e.type AS type, e.description AS description, e.embedding AS embedding
            ORDER BY e.name
            """
        )
        return [self._parse_entity_row(row) for row in result]

    def update_entity(self, name: str, description: str,
                      embedding: list[float] = None) -> bool:
        """更新实体描述和 embedding"""
        result = self._run(
            """
            MATCH (e:Entity {name: $name})
            SET e.description = $description, e.embedding = $embedding
            RETURN count(e) AS n
            """,
            name=name, description=description,
            embedding=json.dumps(embedding) if embedding is not None else None,
        )
        return (result[0]["n"] if result else 0) > 0

    @staticmethod
    def _parse_entity_row(row: dict) -> dict:
        """从展开的 Cypher 返回行解析实体"""
        emb = row.get("embedding")
        return {
            "name": row["name"],
            "type": row.get("type", "") or "",
            "description": row.get("description", "") or "",
            "embedding": json.loads(emb) if emb else None,
        }

    @staticmethod
    def _parse_entity(node) -> dict:
        """兼容旧调用（不再使用，保留备用）"""
        props = {k: v for k, v in node.items()} if hasattr(node, "items") else {}
        emb = props.get("embedding")
        return {
            "name": props["name"],
            "type": props.get("type", ""),
            "description": props.get("description", ""),
            "embedding": json.loads(emb) if emb else None,
        }

    # -------------------- 关系 (Relation) --------------------

    def upsert_relation(self, src: str, tgt: str, keywords: str, description: str,
                        embedding: list[float] = None) -> None:
        """插入或更新关系（同时确保两端实体节点存在）"""
        self._run(
            """
            MERGE (a:Entity {name: $src})
            MERGE (b:Entity {name: $tgt})
            MERGE (a)-[r:RELATION]->(b)
            SET r.keywords = $keywords,
                r.description = $description,
                r.embedding = $embedding
            """,
            src=src, tgt=tgt, keywords=keywords, description=description,
            embedding=json.dumps(embedding) if embedding is not None else None,
        )

    def delete_relation(self, src: str, tgt: str) -> bool:
        """删除指定关系，返回是否实际删除"""
        result = self._run(
            """
            MATCH (a:Entity {name: $src})-[r:RELATION]->(b:Entity {name: $tgt})
            DELETE r RETURN count(r) AS n
            """,
            src=src, tgt=tgt,
        )
        return (result[0]["n"] if result else 0) > 0

    def get_relation(self, src: str, tgt: str) -> Optional[dict]:
        """获取单条关系，返回 {src, tgt, keywords, description, embedding} 或 None"""
        result = self._run(
            """
            MATCH (a:Entity {name: $src})-[r:RELATION]->(b:Entity {name: $tgt})
            RETURN r.keywords AS keywords, r.description AS description, r.embedding AS embedding
            """,
            src=src, tgt=tgt,
        )
        if not result:
            return None
        row = result[0]
        emb = row.get("embedding")
        return {
            "src": src, "tgt": tgt,
            "keywords": row.get("keywords", "") or "",
            "description": row.get("description", "") or "",
            "embedding": json.loads(emb) if emb else None,
        }

    def list_relations(self) -> list[dict]:
        """列出所有关系"""
        result = self._run(
            """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name AS src, b.name AS tgt,
                   r.keywords AS keywords, r.description AS description, r.embedding AS embedding
            """
        )
        return [self._parse_relation_row(row) for row in result]

    def get_relations_by_entity(self, name: str) -> list[dict]:
        """获取与某实体相关的所有关系（出边 + 入边）"""
        result = self._run(
            """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            WHERE a.name = $name OR b.name = $name
            RETURN a.name AS src, b.name AS tgt,
                   r.keywords AS keywords, r.description AS description, r.embedding AS embedding
            """,
            name=name,
        )
        return [self._parse_relation_row(row) for row in result]

    @staticmethod
    def _parse_relation_row(row: dict) -> dict:
        """从展开的 Cypher 返回行解析关系（绕开驱动对象序列化问题）"""
        emb = row.get("embedding")
        return {
            "src": row["src"],
            "tgt": row["tgt"],
            "keywords": row.get("keywords", "") or "",
            "description": row.get("description", "") or "",
            "embedding": json.loads(emb) if emb else None,
        }

    @staticmethod
    def _parse_relation(src: str, tgt: str, rel) -> dict:
        """兼容旧调用（不再使用，保留备用）"""
        props = {k: v for k, v in rel.items()} if hasattr(rel, "items") else {}
        emb = props.get("embedding")
        return {
            "src": src, "tgt": tgt,
            "keywords": props.get("keywords", ""),
            "description": props.get("description", ""),
            "embedding": json.loads(emb) if emb else None,
        }

    # -------------------- 文本块 (Chunk) --------------------

    def upsert_chunk(self, chunk_id: str, content: str, doc_id: str,
                     embedding: list[float] = None) -> None:
        """插入或更新文本块节点"""
        self._run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.content = $content,
                c.doc_id = $doc_id,
                c.embedding = $embedding
            """,
            chunk_id=chunk_id, content=content, doc_id=doc_id,
            embedding=json.dumps(embedding) if embedding is not None else None,
        )

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除单个文本块"""
        result = self._run(
            "MATCH (c:Chunk {chunk_id: $chunk_id}) DELETE c RETURN count(c) AS n",
            chunk_id=chunk_id,
        )
        return (result[0]["n"] if result else 0) > 0

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        """按文档 ID 批量删除文本块，返回删除数量"""
        result = self._run(
            "MATCH (c:Chunk {doc_id: $doc_id}) DELETE c RETURN count(c) AS n",
            doc_id=doc_id,
        )
        return result[0]["n"] if result else 0

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """获取单个文本块"""
        result = self._run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN c.chunk_id AS chunk_id, c.content AS content, c.doc_id AS doc_id, c.embedding AS embedding
            """,
            chunk_id=chunk_id,
        )
        return self._parse_chunk_row(result[0]) if result else None

    def list_chunks(self, doc_id: str = None) -> list[dict]:
        """列出文本块，doc_id 不为空时只返回该文档的块"""
        if doc_id:
            result = self._run(
                """
                MATCH (c:Chunk {doc_id: $doc_id})
                RETURN c.chunk_id AS chunk_id, c.content AS content, c.doc_id AS doc_id, c.embedding AS embedding
                ORDER BY c.chunk_id
                """,
                doc_id=doc_id,
            )
        else:
            result = self._run(
                """
                MATCH (c:Chunk)
                RETURN c.chunk_id AS chunk_id, c.content AS content, c.doc_id AS doc_id, c.embedding AS embedding
                ORDER BY c.chunk_id
                """
            )
        return [self._parse_chunk_row(row) for row in result]

    @staticmethod
    def _parse_chunk_row(row: dict) -> dict:
        """从展开的 Cypher 返回行解析文本块"""
        emb = row.get("embedding")
        return {
            "chunk_id": row["chunk_id"],
            "content": row.get("content", "") or "",
            "doc_id": row.get("doc_id", "") or "",
            "embedding": json.loads(emb) if emb else None,
        }

    @staticmethod
    def _parse_chunk(node) -> dict:
        """兼容旧调用（不再使用，保留备用）"""
        props = {k: v for k, v in node.items()} if hasattr(node, "items") else {}
        emb = props.get("embedding")
        return {
            "chunk_id": props["chunk_id"],
            "content": props.get("content", ""),
            "doc_id": props.get("doc_id", ""),
            "embedding": json.loads(emb) if emb else None,
        }

    # -------------------- Embedding 缓存 --------------------

    def upsert_embedding(self, key: str, embedding: list[float]) -> None:
        """存储独立的 embedding（用于 query 缓存等）"""
        self._run(
            "MERGE (e:EmbCache {key: $key}) SET e.embedding = $embedding",
            key=key, embedding=json.dumps(embedding),
        )

    def get_embedding(self, key: str) -> Optional[list[float]]:
        """获取缓存的 embedding"""
        result = self._run(
            "MATCH (e:EmbCache {key: $key}) RETURN e.embedding AS emb", key=key
        )
        if not result or result[0]["emb"] is None:
            return None
        return json.loads(result[0]["emb"])

    def list_embeddings(self) -> dict[str, list[float]]:
        """返回所有缓存 embedding，格式: {key: vector}"""
        result = self._run("MATCH (e:EmbCache) RETURN e.key AS key, e.embedding AS emb")
        return {row["key"]: json.loads(row["emb"]) for row in result if row["emb"]}

    # -------------------- 统计 --------------------

    def get_stats(self) -> dict:
        """返回各类节点数量统计"""
        def count_nodes(label):
            r = self._run(f"MATCH (n:{label}) RETURN count(n) AS n")
            return r[0]["n"] if r else 0

        def count_rels():
            r = self._run("MATCH ()-[r:RELATION]->() RETURN count(r) AS n")
            return r[0]["n"] if r else 0

        return {
            "entities": count_nodes("Entity"),
            "chunks": count_nodes("Chunk"),
            "relations": count_rels(),
            "embeddings_cached": count_nodes("EmbCache"),
        }

    # -------------------- 清空 --------------------

    def clear_all(self) -> None:
        """删除数据库中所有节点和关系（谨慎使用）"""
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("[Neo4j] 所有数据已清空")
