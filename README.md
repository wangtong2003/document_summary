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

## 技术架构

- **前端**：HTML/CSS/JavaScript
- **后端**：Flask (Python)
- **数据库**：MySQL
- **大语言模型**：Ollama本地部署的开源模型
- **向量存储**：FAISS和ChromaDB
- **文本处理**：PyMuPDF、python-docx、EbookLib等

## 新增功能

- **文本摘要优化**：改进了摘要生成的提示模板，确保生成的摘要更符合用户指定的长度要求
- **摘要格式化**：支持多种摘要格式，包括连续段落、要点式和问答式等
- **独立摘要存储**：添加了独立的摘要存储表，更好地管理纯文本摘要

## 环境配置与安装

### 系统要求

- Python 3.10+
- MySQL 8.0+
- Ollama服务

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

3. 初始化数据库
```bash
mysql -u <username> -p < init.sql
```

4. 配置环境变量
```bash
# 创建.env文件并配置以下变量
FLASK_APP=app.py
DATABASE_URI=mysql+pymysql://username:password@localhost/doc_summary
SECRET_KEY=your-secret-key
```

5. 启动应用
```bash
flask run --host=127.0.0.1 --port=8080
```

## 使用说明

1. 访问 `http://127.0.0.1:8080` 打开应用
2. 上传文档或直接输入文本进行摘要生成
3. 调整摘要参数，包括长度、语言、专业程度等
4. 点击生成按钮获取摘要结果

## 注意事项

- 确保Ollama服务已经启动并加载了所需模型
- 文档大小限制为200MB
- 首次使用大文档时，向量索引创建可能需要较长时间 