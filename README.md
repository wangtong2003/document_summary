# 文档摘要系统 (Document Summary System)

## 项目简介

文档摘要系统是一个基于Flask的Web应用，集成了大语言模型和向量搜索，用于文档处理、摘要生成和语义搜索。系统支持多种文档格式，包括PDF、DOCX、TXT、Markdown和EPUB，能够自动提取文档内容，生成高质量摘要，并提供基于语义的文档检索功能。

## 主要功能

- **文档处理**：支持上传和处理多种格式的文档（PDF、DOCX、TXT、MD、EPUB）
- **智能摘要**：基于大语言模型（Ollama）生成结构化文档摘要
- **主题分析**：自动分析文档主题和关键词
- **语义搜索**：支持基于向量和混合的语义搜索功能
- **摘要管理**：浏览、预览、下载和删除已处理的文档
- **用户管理**：支持用户注册、登录和权限控制（管理员/普通用户）
- **大文件支持**：文件分块存储功能，支持处理大型文档

## 技术架构

- **后端**：Flask, Flask-SQLAlchemy, Flask-JWT-Extended
- **数据库**：MySQL
- **文档处理**：PyMuPDF, python-docx, Markdown, EbookLib
- **AI和向量搜索**：Ollama, langchain, FAISS, sentence-transformers
- **前端**：HTML, CSS, JavaScript, Bootstrap

## 系统要求

- Python 3.8+
- MySQL 5.7+
- Ollama（用于本地运行大语言模型）

## 安装说明

1. 克隆项目仓库

```bash
git clone [仓库地址]
cd document_summary
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 设置数据库

```bash
# 使用MySQL客户端执行init.sql脚本
mysql -u your_username -p < init.sql
```

5. 启动Ollama服务（需要单独安装）

```bash
# 确保Ollama已安装且运行，并加载所需模型
ollama run llama3
```

6. 启动应用

```bash
python app.py
```

应用将在 http://localhost:5000 运行

## 使用指南

### 用户登录

- 访问登录页面：`/login`
- 默认管理员账户：
  - 用户名：admin
  - 密码：admin123

### 文档处理

1. 访问主页或仪表板
2. 上传文档文件（支持PDF、DOCX、TXT、MD、EPUB）
3. 选择摘要长度和目标语言
4. 点击"处理"按钮
5. 系统将自动处理文档并生成摘要

### 摘要管理

- 在摘要库页面可以查看所有已处理的文档
- 支持预览、下载和删除操作
- 支持基于关键词的搜索功能

### 语义搜索

- 在文档详情页面可以使用语义搜索功能
- 支持多种搜索模式：语义搜索、混合搜索
- 支持生成针对特定问题的语义摘要

### 用户管理（管理员功能）

- 管理员可以访问用户管理页面：`/admin/users`
- 支持创建、查看、编辑和删除用户

## 文件结构

```
document_summary/
├── app.py                # 主应用程序
├── requirements.txt      # 项目依赖
├── init.sql              # 数据库初始化脚本
├── templates/            # HTML模板
│   ├── admin/            # 管理员页面模板
│   ├── dashboard.html    # 仪表板页面
│   ├── login.html        # 登录页面
│   ├── register.html     # 注册页面
│   ├── summaries.html    # 摘要列表页面
│   └── preview.html      # 文档预览页面
├── static/               # 静态资源
├── uploads/              # 上传文件临时存储
└── documents/            # 文档存储目录
```

## 注意事项

- 系统依赖Ollama服务进行摘要生成，请确保Ollama已正确安装并运行
- 处理大型文档可能需要较长时间，系统采用分块处理和存储方式提高效率
- 默认管理员账户在首次启动应用时自动创建，请及时修改默认密码

## 授权许可

[项目授权信息] 