# -*- coding: utf-8 -*-
"""
GraphRAGLite + Neo4j 存储后端使用示例

前置条件:
  1. 启动 Neo4j（Docker 一键启动）:
       docker run -d --name neo4j \
         -p 7474:7474 -p 7687:7687 \
         -e NEO4J_AUTH=neo4j/your_password \
         neo4j:5
  2. 安装依赖: pip install neo4j graphrag-lite
  3. 配置 .env 文件（参考项目根目录 .env）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_lite import GraphRAGLite
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

SAMPLE_TEXT = """
《三国演义》是中国古典四大名著之一。

曹操是东汉末年著名的政治家、军事家，挟天子以令诸侯，统一北方。
他善用谋士，麾下有郭嘉、荀彧、贾诩等人。

诸葛亮字孔明，号卧龙，蜀汉丞相。他运筹帷幄，辅佐刘备建立蜀汉政权。
著名的空城计、草船借箭均出自其手。

刘备是汉室宗亲，仁义著称，与关羽、张飞桃园三结义。
他三顾茅庐请出诸葛亮，最终建立蜀汉。

关羽字云长，忠义无双，使青龙偃月刀，被后世尊为武圣。
"""


def main():
    print("=" * 60)
    print("GraphRAGLite + Neo4j 示例")
    print("=" * 60)

    # ── 初始化（传入 neo4j_* 参数即启用 Neo4j 后端）──
    graph = GraphRAGLite(
        storage_path="./tmp/graphrag_neo4j",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model="qwen-max",
        embedding_model="text-embedding-v3",
        enable_cache=True,
        # Neo4j 配置
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )

    # ── 插入数据 ──
    if not graph.has_data():
        print("\n[1] 插入文档，构建知识图谱...")
        result = graph.insert(SAMPLE_TEXT, doc_id="三国演义")
        print(f"    插入结果: {result}")
    else:
        print("\n[1] 已有数据，跳过插入")

    # ── 统计 ──
    stats = graph.get_stats()
    print(f"\n[2] 统计: {stats}")
    print(f"    实体列表: {graph.list_entities()}")
    print(f"    关系列表: {graph.list_relations()}")

    # ── 查询 ──
    print("\n[3] 查询测试")
    print("-" * 50)
    q = "诸葛亮和刘备是什么关系？"
    print(f"问题: {q}")
    for mode in ["local", "global", "mix"]:
        ans = graph.query(q, mode=mode, top_k=5)
        print(f"  [{mode:6}] {ans[:120]}..." if len(ans) > 120 else f"  [{mode:6}] {ans}")

    # ── Neo4j 直接增删改查 ──
    print("\n[4] Neo4j 增删改查 API 演示")
    print("-" * 50)

    # 新增实体
    print("  新增实体: 司马懿")
    graph.neo4j_add_entity(
        name="司马懿",
        entity_type="人物",
        description="曹魏重臣，善用兵法，后其孙司马炎建立西晋。",
    )
    print(f"  实体列表: {graph.list_entities()}")

    # 更新实体
    print("  更新实体: 司马懿")
    graph.neo4j_update_entity(
        name="司马懿",
        description="曹魏四朝重臣，与诸葛亮多次对阵，以坚守著称。其孙司马炎建立西晋。",
    )

    # 新增关系
    print("  新增关系: 司马懿 -> 曹操")
    graph.neo4j_add_relation(
        src="司马懿",
        tgt="曹操",
        keywords="效忠 臣属",
        description="司马懿早年效忠曹操，是曹魏重要谋臣。",
    )
    print(f"  关系列表: {graph.list_relations()}")

    # 查询 Neo4j 底层 API
    neo = graph._neo4j
    print(f"  Neo4j 直接查实体 '诸葛亮': {neo.get_entity('诸葛亮')}")
    print(f"  Neo4j 统计: {neo.get_stats()}")

    # 删除关系
    # print("  删除关系: 司马懿 -> 曹操")
    # graph.neo4j_delete_relation("司马懿", "曹操")

    # # 删除实体
    # print("  删除实体: 司马懿")
    # graph.neo4j_delete_entity("司马懿")
    # print(f"  实体列表: {graph.list_entities()}")

    print("\n完成！")


if __name__ == "__main__":
    main()
