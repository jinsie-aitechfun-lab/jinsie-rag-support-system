# Retrieval Modes

系统支持两种检索模式：

- keyword：关键词匹配（字符串包含/命中评分）
- vector：向量语义相似度检索（embedding + cosine similarity）

区别：

- keyword 更快、更简单，但依赖精确词命中
- vector 可以理解语义，即使换说法也能召回
- 在真实企业知识库中，vector 更适合自然语言问答
