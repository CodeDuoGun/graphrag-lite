# neo install
```bash
pip install neo4j
# 或
pip install graphrag-lite[neo4j]
```


# neo start
```bash
docker run --rm --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/graphadmin neo4j:community-ubi10

```
