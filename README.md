# 文档摘要助手

基于大语言模型的智能文档摘要与分析系统，支持多种文档格式，提供高质量摘要生成与文档交互功能。

## 主要功能

- **多格式文档支持**：支持PDF、Word文档、TXT、Markdown和EPUB等多种格式
- **文本摘要生成**：利用大语言模型自动生成高质量摘要
- **自定义摘要选项**：支持调整摘要长度、目标语言、专业程度和风格
- **关键词提取**：自动从文档中提取关键词
- **语义搜索**：基于向量数据库的高效文档内容搜索
- **RAG问答**：与文档进行自然语言对话交互
- **主题分析**：自动分析文档的主题结构
- **GPU加速**：支持使用GPU加速向量检索操作
- **用户管理系统**：完整的用户注册、登录和权限管理
- **文档库管理**：集中管理和组织已处理的文档
- **流式处理**：支持大型文档的流式处理和响应

## 技术架构

- **前端**：HTML/CSS/JavaScript
- **后端**：Flask (Python)
- **数据库**：MySQL（使用SQLAlchemy ORM）
- **大语言模型**：Ollama本地部署的开源模型
- **向量存储**：FAISS和ChromaDB
- **向量嵌入**：本地部署的Ollama嵌入模型（snowflake-arctic-embed2）
- **文本处理**：PyMuPDF、python-docx、EbookLib等
- **身份验证**：Flask-Session用于会话管理

## 高级功能

- **文本摘要优化**：改进了摘要生成的提示模板，确保生成的摘要更符合用户指定的长度要求
- **摘要格式化**：支持多种摘要格式，包括连续段落、要点式和问答式等
- **独立摘要存储**：添加了独立的摘要存储表，更好地管理纯文本摘要
- **本地嵌入优化**：优化使用本地Ollama模型进行文本嵌入，提高处理速度
- **智能分块策略优化**：改进文本分块策略，支持动态块大小和重叠率，根据文档类型和长度自动调整，提高检索质量
- **高级混合语义搜索**：增强混合语义搜索算法，添加特殊术语识别、权重调整和滑动窗口评估，提升检索结果的相关性和连贯性
- **连续性保障**：通过相邻文本块评分增强，确保检索结果的上下文连贯性，避免片段化理解
- **自适应权重分配**：为不同类型的内容（原文与摘要）自动分配权重，提高混合搜索的准确性
- **BM25排序算法**：集成传统的BM25算法与向量检索，提高搜索结果的精准度
- **大文件分块处理**：采用智能分块技术处理超大文件，避免内存溢出
- **异常重试机制**：针对数据库操作实现自动重试机制，提高系统稳定性

## 数据模型

系统包含以下主要数据模型：

- **User**：用户信息管理，包括用户名、密码、邮箱和角色
- **DocumentSummary**：文档摘要主表，存储文件信息、摘要内容和向量数据
- **FileChunk**：文件分块存储，用于处理大型文档
- **FileMapping**：文件映射关系记录
- **Summary**：独立摘要记录，用于纯文本摘要管理

## 环境配置与安装

### 系统要求

- Python 3.10+
- MySQL 8.0+
- Ollama服务 (用于LLM和文本嵌入)
- (可选) NVIDIA GPU，用于加速向量操作

### 安装步骤

1. 克隆代码库
```bash
git clone <repository-url>
cd document_summary
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装GPU支持（可选）
```bash
# 如果有NVIDIA GPU并想启用GPU加速，请运行:
python install_gpu.py
```

4. 安装必要的Ollama模型
```bash
# 确保已安装Ollama并下载嵌入模型
ollama pull snowflake-arctic-embed2
```

5. 初始化数据库
```bash
mysql -u <username> -p < init.sql
```

6. 配置环境变量
```bash
# 创建.env文件并配置以下变量
FLASK_APP=app.py
DATABASE_URI=mysql+pymysql://username:password@localhost/doc_summary
SECRET_KEY=your-secret-key
```

7. 启动应用
```bash
flask run --host=127.0.0.1 --port=8080
```

## 使用说明

1. 访问 `http://127.0.0.1:8080` 打开应用
2. 注册账号或使用默认管理员账号登录（如已配置）
3. 上传文档或直接输入文本进行摘要生成
4. 调整摘要参数，包括长度、语言、专业程度等
5. 点击生成按钮获取摘要结果
6. 使用语义搜索功能在文档中查找信息
7. 通过RAG交互功能与文档进行对话

## 高级用法

### RAG问答系统

系统的RAG工具提供了强大的文档问答能力：

```python
# 示例API调用
POST /semantic_search/<doc_id>
{
    "query": "文档中关于技术架构的内容是什么？"
}
```

### 混合语义搜索

混合搜索结合了内容和摘要的向量表示，提供更精准的搜索结果：

```python
# 示例API调用
POST /hybrid_search/<doc_id>
{
    "query": "搜索关键词",
    "content_weight": 0.6,
    "summary_weight": 0.4
}
```

## GPU加速说明

系统默认会检测是否有可用的GPU，如果有将自动使用GPU进行向量检索操作，显著提高文档处理和搜索性能。

- 要验证GPU是否正常工作，可以运行:
```bash
python -c "import torch; print('CUDA可用:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

- 如果有GPU但未被正确检测，请确保已安装正确版本的CUDA和PyTorch

## API参考

系统提供以下主要API端点：

- `/process_document`: 处理上传文档并生成摘要
- `/process_document_stream`: 流式处理文档并返回实时进度
- `/semantic_search/<doc_id>`: 对指定文档执行语义搜索
- `/hybrid_search/<doc_id>`: 对指定文档执行混合语义搜索
- `/semantic_summary/<doc_id>`: 根据查询生成定向摘要
- `/api/search`: 全局搜索所有文档
- `/summaries`: 获取所有摘要列表
- `/summaries/<summary_id>`: 获取、删除特定摘要

## 注意事项

- 确保Ollama服务已经启动并加载了所需模型
- 文档大小限制为200MB
- 首次使用大文档时，向量索引创建可能需要较长时间
- GPU加速需要安装兼容版本的PyTorch和CUDA
- 系统已配置过滤大多数LangChain和Pydantic警告消息，提供更清晰的日志输出

## 维护与故障排除

- 如遇到数据库连接问题，系统会自动尝试重试操作（最多3次）
- 向量存储问题可通过`/api/reinit_rag_tools`重新初始化RAG工具
- 所有文档的向量索引可通过`/api/reindex_all`重建
- 管理员可通过`/admin/users`管理用户账户 