# Vector Retrieval Pipeline

向量召回流程：

1. 将文档切分为 chunk
2. 使用 embedding 模型生成向量
3. 存入 Chroma 向量数据库
4. 查询时将问题向量化
5. 通过相似度搜索返回最相关片段

本项目使用：

- Embedding API（OpenAI-compatible）
- Chroma persistent index（var/chroma）
- cosine 相似度
