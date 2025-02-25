import pymysql
pymysql.install_as_MySQLdb()

from flask import Flask, request, jsonify, Response, render_template, stream_with_context, send_file, make_response, session, send_from_directory
from flask_session import Session  # 添加 Flask-Session 导入
import asyncio
from ollama import Client
import os
import fitz  # PyMuPDF
from docx import Document
import markdown
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import mysql.connector
import uuid
from datetime import datetime, timedelta
import jwt
from functools import wraps
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager, jwt_required, get_jwt_identity, create_access_token,
    exceptions as jwt_exceptions
)
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import shutil
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_migrate import Migrate
from io import BytesIO
from urllib.parse import quote
import traceback
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import string
import PyPDF2  # 添加 PyPDF2 导入
import time
import sqlalchemy.exc
from sqlalchemy import inspect  # 修改为从sqlalchemy直接导入inspect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Any
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import copy

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DOCUMENTS_FOLDER = 'documents'  # 新增永久文档存储目录
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md', 'epub'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOCUMENTS_FOLDER'] = DOCUMENTS_FOLDER  # 新增配置
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 修改为 200MB
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf8mb4'

# 添加分块上传的配置
app.config['MAX_CONTENT_LENGTH'] = None  # 禁用全局限制
app.config['MAX_FILE_SIZE'] = 200 * 1024 * 1024  # 修改为 200MB 单文件限制

# MySQL配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/document_summary?charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,  # 连接池大小
    'pool_recycle': 3600,  # 连接回收时间(秒)
    'pool_pre_ping': True,  # 自动检测连接是否有效
    'pool_timeout': 30,  # 连接池获取连接的超时时间
    'max_overflow': 5,  # 连接池最大溢出连接数
    'connect_args': {
        'connect_timeout': 60,  # 连接超时时间
        'read_timeout': 60,  # 读取超时时间
        'write_timeout': 60,  # 写入超时时间
        'charset': 'utf8mb4',
        'init_command': "SET time_zone='+00:00'",  # 设置时区
        'autocommit': True  # 自动提交
    },
    'execution_options': {
        'pool_timeout': 30,
        'pool_pre_ping': True,
        'isolation_level': 'READ COMMITTED'  # 设置事务隔离级别
    }
}

# Session 配置
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 设置 session 密钥
app.config['SESSION_TYPE'] = 'filesystem'  # 使用文件系统存储 session
app.config['SESSION_FILE_DIR'] = 'flask_session'  # session 文件存储目录
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # session 有效期

# 初始化 Flask-Session
Session(app)

db = SQLAlchemy(app)
migrate = Migrate(app, db)  # 初始化 Flask-Migrate

# 创建必要的目录
for folder in [UPLOAD_FOLDER, DOCUMENTS_FOLDER, 'flask_session']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 数据模型定义
class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 添加自增属性
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    
    def __repr__(self):
        return f'<User {self.username}>'

class FileChunk(db.Model):
    __tablename__ = 'file_chunks'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document_summaries.id', ondelete='CASCADE'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    chunk_data = db.Column(db.LargeBinary(length=10 * 1024 * 1024), nullable=False)  # 10MB per chunk
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_engine': 'InnoDB',
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }

class DocumentSummary(db.Model):
    __tablename__ = 'document_summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(32), nullable=True)
    original_text = db.Column(db.Text(length=16777215), nullable=True)
    summary_text = db.Column(db.Text(length=16777215), nullable=True)
    file_content = db.Column(db.LargeBinary(length=16777215), nullable=True)
    content_vectors = db.Column(db.LargeBinary(length=16777215), nullable=True)
    summary_vectors = db.Column(db.LargeBinary(length=16777215), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary_length = db.Column(db.String(20))
    target_language = db.Column(db.String(20))
    file_size = db.Column(db.BigInteger)
    mime_type = db.Column(db.String(100))
    original_filename = db.Column(db.String(255))
    display_filename = db.Column(db.String(255))
    keywords = db.Column(db.String(255))  # 增加 keywords 字段长度到 255
    topic_analysis = db.Column(db.JSON)
    embedding_model = db.Column(db.String(100))
    chunks_info = db.Column(db.JSON)
    is_chunked = db.Column(db.Boolean, default=False)  # 添加是否分块存储标志
    total_chunks = db.Column(db.Integer, default=0)    # 添加总块数字段
    
    # 关联文件块
    chunks = db.relationship('FileChunk', backref='document', lazy='dynamic',
                           cascade='all, delete-orphan')
    
    __table_args__ = {
        'mysql_engine': 'InnoDB',
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }

class FileMapping(db.Model):
    __tablename__ = 'file_mappings'
    
    id = db.Column(db.Integer, primary_key=True)
    summary_id = db.Column(db.Integer, db.ForeignKey('document_summaries.id', ondelete='CASCADE'), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    system_filename = db.Column(db.String(255), nullable=False)
    display_filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    summary = db.relationship('DocumentSummary', backref=db.backref('file_mappings', lazy=True))

# 添加数据库连接重试装饰器
def retry_on_db_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except sqlalchemy.exc.OperationalError as e:
                    if "Lost connection" in str(e) and retries < max_retries - 1:
                        retries += 1
                        print(f"数据库连接丢失,尝试第{retries}次重连...")
                        time.sleep(delay)
                        # 重新初始化数据库连接
                        db.session.remove()
                        continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 为关键数据库操作添加重试机制
@retry_on_db_error()
def save_file_content(summary, file_content):
    """保存文件内容,根据大小决定是否分块存储"""
    if file_content is None:
        return
        
    file_size = len(file_content)
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        # 确保summary对象绑定到当前session
        if not db.session.is_active:
            db.session.begin()
        
        # 如果对象是detached状态，重新merge到session
        if inspect(summary).detached:
            summary = db.session.merge(summary)
        
        # 删除现有的chunks(如果有的话)
        FileChunk.query.filter_by(document_id=summary.id).delete()
        
        # 小文件(<=10MB)直接存储在主表中
        if file_size <= CHUNK_SIZE:
            summary.file_content = file_content
            summary.file_size = file_size
            db.session.commit()
            return
            
        # 大文件分块存储
        summary.file_content = None  # 清空主表的file_content字段
        summary.file_size = file_size
        db.session.commit()
        
        # 计算需要的块数
        chunk_count = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        summary.total_chunks = chunk_count
        summary.is_chunked = True
        db.session.commit()
        
        # 分块存储
        for i in range(chunk_count):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, file_size)
            chunk_data = file_content[start:end]
            
            chunk = FileChunk(
                document_id=summary.id,
                chunk_index=i,
                chunk_data=chunk_data
            )
            db.session.add(chunk)
            
            # 每20个块提交一次，避免事务太大
            if (i + 1) % 20 == 0 or i == chunk_count - 1:
                try:
                    db.session.commit()
                except Exception as e:
                    print(f"提交第 {i+1} 个块时出错: {str(e)}")
                    db.session.rollback()
                    raise
        
    except Exception as e:
        print(f"保存文件内容时出错: {str(e)}")
        if db.session.is_active:
            db.session.rollback()
        raise

def get_file_content(summary_id):
    """获取文件内容，自动处理分块存储的情况"""
    summary = DocumentSummary.query.get(summary_id)
    if not summary:
        return None
        
    # 如果文件内容直接存储在主表中
    if summary.file_content is not None:
        return summary.file_content
        
    # 获取所有分块并按顺序拼接
    chunks = FileChunk.query.filter_by(document_id=summary_id).order_by(FileChunk.chunk_index).all()
    if not chunks:
        return None
        
    return b''.join(chunk.chunk_data for chunk in chunks)

@retry_on_db_error()
def save_summary_to_db(file_info, summary_text, params, file_content=None):
    """保存或更新文档摘要到MySQL"""
    try:
        print("\n=== 开始保存摘要到数据库 ===")
        print(f"摘要长度: {len(summary_text) if summary_text else 0}")
        print(f"参数: {params}")
        
        # 确保session处于清洁状态
        db.session.rollback()
        
        # 计算文件内容的MD5哈希值
        import hashlib
        file_hash = hashlib.md5(file_info["original_text"].encode('utf-8')).hexdigest()
        
        # 获取原始文件名和显示文件名
        original_filename = file_info.get("original_filename")
        display_filename = original_filename
        
        print(f"原始文件名: {original_filename}")
        print(f"显示文件名: {display_filename}")

        try:
            # 检查是否已存在相同文件的摘要
            existing_summary = DocumentSummary.query.filter_by(
                file_hash=file_hash
            ).first()
            
            if existing_summary:
                print(f"更新现有摘要 ID: {existing_summary.id}")
                # 如果对象是detached状态，重新merge到session
                if inspect(existing_summary).detached:
                    existing_summary = db.session.merge(existing_summary)
                
                # 更新现有摘要
                existing_summary.summary_text = summary_text
                existing_summary.summary_length = params.get("summary_length")
                existing_summary.target_language = params.get("target_language")
                existing_summary.original_filename = original_filename
                existing_summary.display_filename = display_filename
                existing_summary.keywords = file_info.get('keywords')
                existing_summary.topic_analysis = file_info.get('topic_analysis')
                
                try:
                    db.session.commit()
                except Exception as e:
                    print(f"提交摘要更新时出错: {str(e)}")
                    db.session.rollback()
                    raise
                
                if file_content:
                    save_file_content(existing_summary, file_content)
                    existing_summary.file_size = len(file_content)
                    existing_summary.mime_type = file_info.get('mime_type')
                    db.session.commit()
                
                existing_summary.updated_at = datetime.now()
                db.session.commit()
                
                # 创建或更新混合向量存储
                try:
                    create_hybrid_vector_store(file_info["original_text"], summary_text, existing_summary.id)
                except Exception as e:
                    print(f"创建混合向量存储失败: {str(e)}")
                    # 继续处理，不影响主流程
                
                # 更新文件名映射
                file_mapping = FileMapping.query.filter_by(summary_id=existing_summary.id).first()
                if file_mapping:
                    if inspect(file_mapping).detached:
                        file_mapping = db.session.merge(file_mapping)
                    file_mapping.original_filename = original_filename
                    file_mapping.system_filename = file_info["filename"]
                    file_mapping.display_filename = display_filename
                else:
                    new_mapping = FileMapping(
                        summary_id=existing_summary.id,
                        original_filename=original_filename,
                        system_filename=file_info["filename"],
                        display_filename=display_filename
                    )
                    db.session.add(new_mapping)
                db.session.commit()
                print("摘要更新成功")
            else:
                print("创建新摘要记录")
                # 创建新摘要记录
                new_summary = DocumentSummary(
                    file_name=file_info["filename"],
                    file_hash=file_hash,
                    summary_text=summary_text,
                    original_text=file_info["original_text"],
                    summary_length=params.get("summary_length"),
                    target_language=params.get("target_language"),
                    file_size=len(file_content) if file_content else None,
                    mime_type=file_info.get('mime_type'),
                    original_filename=original_filename,
                    display_filename=display_filename,
                    keywords=file_info.get('keywords'),
                    topic_analysis=file_info.get('topic_analysis')
                )
                db.session.add(new_summary)
                
                try:
                    db.session.commit()
                except Exception as e:
                    print(f"提交新摘要记录时出错: {str(e)}")
                    db.session.rollback()
                    raise
                
                # 保存文件内容
                if file_content:
                    save_file_content(new_summary, file_content)
                
                # 创建混合向量存储
                try:
                    print(f"原始文本类型: {type(new_summary.original_text)}, 原始文本长度: {len(new_summary.original_text) if new_summary.original_text else 0}")
                    print(f"摘要文本类型: {type(new_summary.summary_text)}, 摘要文本长度: {len(new_summary.summary_text) if new_summary.summary_text else 0}")
                    create_hybrid_vector_store(
                        new_summary.original_text,
                        new_summary.summary_text,
                        new_summary.id
                    )
                    print("向量存储创建成功")
                except Exception as e:
                    print(f"创建向量存储失败: {str(e)}")
                    # 继续处理，不要因为向量存储失败而影响整个摘要保存
                    pass
                
                # 创建文件名映射
                new_mapping = FileMapping(
                    summary_id=new_summary.id,
                    original_filename=original_filename,
                    system_filename=file_info["filename"],
                    display_filename=display_filename
                )
                db.session.add(new_mapping)
                db.session.commit()
                print("摘要保存成功")
            
            return True
            
        except sqlalchemy.exc.OperationalError as e:
            print(f"数据库操作错误: {str(e)}")
            db.session.rollback()
            raise
        except Exception as e:
            print(f"数据库操作出错: {str(e)}")
            print(f"完整错误信息: {e.__class__.__name__}: {str(e)}")
            traceback.print_exc()
            db.session.rollback()
            raise
            
    except Exception as e:
        print(f"保存摘要到数据库时出错: {str(e)}")
        print(f"完整错误信息: {e.__class__.__name__}: {str(e)}")
        traceback.print_exc()
        db.session.rollback()
        raise

# 文档处理函数
def read_pdf(file_path):
    """读取PDF文件内容"""
    try:
        print(f"开始读取PDF文件: {file_path}")
        text = ""
        
        # 首先尝试使用 PyMuPDF (更稳定的选择)
        try:
            with fitz.open(file_path) as doc:
                num_pages = doc.page_count
                print(f"PDF文件共有 {num_pages} 页 (PyMuPDF)")
                
                # 创建线程池
                from concurrent.futures import ThreadPoolExecutor
                from threading import Lock
                
                # 创建一个锁用于同步写入
                text_lock = Lock()
                page_texts = [""] * num_pages  # 预分配页面文本列表
                
                def process_page_fitz(page_num):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            print(f"成功读取第 {page_num + 1} 页，提取到 {len(page_text)} 个字符")
                            return page_num, page_text + "\n\n"
                        return page_num, ""
                    except Exception as e:
                        print(f"PyMuPDF读取第 {page_num + 1} 页时出错: {str(e)}")
                        return page_num, ""
                
                # 使用线程池并行处理页面
                with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                    futures = [executor.submit(process_page_fitz, i) for i in range(num_pages)]
                    for future in futures:
                        page_num, page_text = future.result()
                        page_texts[page_num] = page_text
                
                # 合并所有页面文本
                text = "".join(page_texts)
        except Exception as e:
            print(f"使用PyMuPDF读取PDF失败: {str(e)}")
            # 如果PyMuPDF失败，可以在这里添加备用的PDF读取方法
            raise
        
        if not text.strip():
            raise Exception("未能从PDF中提取任何文本")
            
        print(f"PDF文件读取完成，总共提取到 {len(text)} 个字符")
        return text
        
    except Exception as e:
        print(f"读取PDF文件出错: {str(e)}")
        raise Exception(f"读取PDF文件失败: {str(e)}")

def read_docx(file_path):
    """读取Word文档内容"""
    try:
        print(f"开始读取Word文档: {file_path}")
        
        # 确保文件存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 检查文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.doc':
            # 对于旧版.doc文件，尝试使用其他方法读取
            try:
                import win32com.client
                word = win32com.client.Dispatch("Word.Application")
                word.Visible = False
                doc = word.Documents.Open(os.path.abspath(file_path))
                text = doc.Content.Text
                doc.Close()
                word.Quit()
                return text
            except Exception as e:
                print(f"使用win32com读取.doc文件失败: {str(e)}")
                # 如果win32com失败，返回提示信息
                return "无法读取旧版Word文档(.doc)，请将文档另存为.docx格式后重试。"
        
        # 对于.docx文件使用python-docx
        doc = Document(file_path)
        text_parts = []
        
        # 创建线程池
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock
        
        # 创建一个锁用于同步写入
        text_lock = Lock()
        
        def process_paragraph(para):
            if para.text.strip():
                return para.text + "\n"
            return ""
            
        def process_table(table):
            table_text = []
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        table_text.append(cell.text)
            return "\n".join(table_text) + "\n" if table_text else ""
        
        # 使用线程池并行处理段落和表格
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            # 并行处理段落
            para_futures = [executor.submit(process_paragraph, para) for para in doc.paragraphs]
            # 并行处理表格
            table_futures = [executor.submit(process_table, table) for table in doc.tables]
            
            # 收集段落结果
            for future in para_futures:
                text = future.result()
                if text:
                    text_parts.append(text)
                    
            # 收集表格结果
            for future in table_futures:
                text = future.result()
                if text:
                    text_parts.append(text)
        
        # 合并所有文本
        result = "".join(text_parts)
        
        if not result.strip():
            raise ValueError("文档内容为空")
            
        print(f"成功读取Word文档，提取到 {len(result)} 个字符")
        return result
        
    except Exception as e:
        print(f"读取Word文档时出错: {str(e)}")
        raise Exception(f"读取Word文档失败: {str(e)}")

def read_txt(file_path):
    """读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试其他编码
        encodings = ['gbk', 'gb2312', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        raise Exception("无法识别文件编码")

def read_md(file_path):
    """读取 Markdown 文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return markdown.markdown(text)
    except UnicodeDecodeError:
        # 果 UTF-8 失败，尝试其他编码
        encodings = ['gbk', 'gb2312', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                return markdown.markdown(text)
            except UnicodeDecodeError:
                continue
        raise Exception("无法识别文编码")

def read_epub(file_path):
    book = epub.read_epub(file_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text += soup.get_text() + '\n'
    return text

def read_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:  # 同时支持 .docx 和 .doc
        return read_docx(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    elif file_extension == '.md':
        return read_md(file_path)
    elif file_extension == '.epub':
        return read_epub(file_path)
    else:
        raise ValueError("不支持的文件格式")

def generate_summary(text):
    """生成文档摘要"""
    try:
        print("开始生成摘要...")
        print(f"输入文本长度: {len(text)} 字符")
        
        client = Client(host='http://localhost:11434')
        
        # 构建提示词
        system_prompt = """你是一个专业的文档摘要和关键词提取专家。请按照以下思维步骤分析文档并生成高质量摘要：

        [思考步骤]
        1. 分析：仔细阅读整个文档，确定文档的主题、目的和主要论点
        2. 提取：识别文档中的核心概念、关键信息点和重要论述
        3. 判断类型：确定文档是学术论文、研究报告、技术文档还是一般文章
        4. 分类：
           - 对于学术/研究类文档：按"研究背景"、"研究问题"、"方法论"、"主要发现"、"研究意义"分类
           - 对于技术/一般类文档：按"引言"、"核心论点"、"方法/发现"、"结论"分类
        5. 提炼：从每个类别中提取最具代表性的内容，确保覆盖文档的实质
        6. 关键词：确定能够准确代表文档核心内容的4个最重要关键词
        7. 整合：将以上信息整合成连贯、简洁的摘要，保持原文的核心意义

        [输出格式要求]
        1. 首先输出 [KEYWORDS] 标记
        2. 在其下方输出4个最重要的关键词，用竖线(|)分隔
        3. 然后输出 [SUMMARY] 标记
        4. 最后输出结构化摘要正文，根据文档类型选择适当的结构：
           
           学术/研究类文档结构：
           - 引言（约75字）：简述研究背景和目的
           - 核心论点（约200字）：按条理列出研究的主要理论观点
           - 研究方法（约100字）：描述研究采用的方法和数据
           - 研究发现（约100字）：总结主要的研究结果或发现
           - 结论与意义（约75字）：指出研究的结论和实践意义
           
           技术/一般类文档结构：
           - 引言（约75字）：简述文档背景和目的
           - 核心论点（约250字）：按条理列出文档的主要观点
           - 方法/发现（约100字）：描述文档中的关键方法或发现
           - 结论（约75字）：总结文档的结论或展望

        [关键词要求]
        1. 必须提取4个关键词
        2. 关键词应该是文档中最具代表性的词语
        3. 每个关键词长度建议在2-4个字之间
        4. 关键词之间使用竖线(|)分隔，不要有多余的空格
        5. 关键词要按重要性排序

        [摘要要求]
        1. 摘要总长度控制在500字左右
        2. 使用清晰的段落结构，每个主要部分单独成段
        3. 为每个段落添加明确的小标题（如"引言："、"核心论点："等）
        4. 保持客观性和准确性，不添加原文中不存在的内容
        5. 对于学术文献，保留核心术语和引用信息（如"研究[1]表明..."）
        6. 使用清晰流畅的语言，对专业术语给予必要解释
        7. 确保摘要能独立于原文被理解

        示例输出格式（学术论文）：
        [KEYWORDS]
        大语言模型|文本摘要|思维链|效果评估

        [SUMMARY]
        引言：本研究探讨了思维链方法在提升大语言模型文档摘要能力方面的应用。随着大语言模型在自然语言处理领域的广泛应用，提高其在文档摘要任务中的表现变得尤为重要。

        核心论点：
        1. 大语言模型（如GPT系列）虽具强大的文本理解和生成能力，但在文档摘要中表现出色，能够减少人工干预的需求[2]。研究表明，使用预训练的语言模型并对其进行微调可以有效提升文档摘要质量。
        2. 思维链方法通过引导模型按步骤思考，显著改善了摘要的结构性和准确性。这种方法在实验中获得了比传统摘要方法高15%的用户满意度[3]。
        3. 国内外研究均表明大语言模型在文档摘要领域有广泛应用。国外方面，冯志伟总结了大语言模型不仅促进了自然语言处理技术的工程成功，还深刻改变了语言知识的生产方式[5]。

        研究方法：本研究采用了对比实验方法，比较了普通提示词和思维链提示词生成的摘要质量。评估维度包括信息完整性、结构清晰度和关键信息提取能力。研究使用了标准摘要数据集进行测试，并通过人工评估验证结果。

        研究发现：实验结果表明，基于思维链方法的提示策略能够显著提高摘要的质量和可用性。特别是在结构化信息提取和关键观点识别方面，思维链方法的表现优于传统方法。用户反馈也显示，思维链生成的摘要更易于理解和使用。

        结论与意义：思维链方法为提升大语言模型在文档摘要任务中的表现提供了有效途径。这一发现不仅对改进文档摘要技术具有重要意义，也为其他自然语言处理任务提供了参考。未来研究将进一步探索思维链方法在更复杂文档类型中的适用性。
        """
        
        content = f"""
        需要分析的文本内容：
        <文本>
        {text}
        </文本>

        请按照思维步骤分析文档，首先判断文档类型（学术论文、研究报告、技术文档或一般文章），然后基于文档类型选择适当的摘要结构，按规定格式提取关键词并生成结构化摘要。确保输出的关键词正好是4个，用竖线分隔。
        """
        
        print("调用 Ollama API...")
        response = client.generate(
            model='huihui_ai/qwen2.5-1m-abliterated:latest',
            prompt=system_prompt + content,
            stream=False
        )
        
        if not response or 'response' not in response:
            raise Exception("API 返回的数据格式不正确")
            
        summary = response['response']
        print(f"成功生成摘要，长度: {len(summary)} 字符")
        return summary
        
    except Exception as e:
        print(f"生成摘要时发生错误: {str(e)}")
        raise Exception(f"生成摘要失败: {str(e)}")

def get_file_mime_type(filename):
    """获取文件的MIME类型"""
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.epub': 'application/epub+zip'
    }
    ext = os.path.splitext(filename)[1].lower()
    return mime_types.get(ext, 'application/octet-stream')

@app.route('/process_document', methods=['POST'])
def process_document():
    """处理上传的文档并生成摘要"""
    try:
        print("\n=== 开始处理文档 ===")
        
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        files = request.files.getlist('file')  # 获取所有上传的文件
        if not files or files[0].filename == '':
            return jsonify({'error': '未选择文件'}), 400
            
        # 确保上传目录存在
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        # 获取参数
        params = {
            'summary_length': request.form.get('summary_length', 'medium'),
            'target_language': request.form.get('target_language', 'chinese'),
            'summary_style': request.form.get('summary_style'),
            'focus_area': request.form.get('focus_area'),
            'expertise_level': request.form.get('expertise_level'),
            'language_style': request.form.get('language_style')
        }
        print("接收到的参数:", params)
        
        results = []  # 存储所有文件的处理结果
        
        for file in files:
            try:
                # 检查文件大小
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > app.config['MAX_FILE_SIZE']:
                    results.append({
                        'filename': file.filename,
                        'error': f'文件大小超过限制，最大允许 {app.config["MAX_FILE_SIZE"] // (1024 * 1024)}MB'
                    })
                    continue
                
                # 保存原始文件名
                original_filename = file.filename
                print(f"处理文件: {original_filename}")
                
                # 检查文件类型
                if not allowed_file(original_filename):
                    results.append({
                        'filename': original_filename,
                        'error': '不支持的文件格式'
                    })
                    continue
                
                # 生成安全的文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                _, ext = os.path.splitext(original_filename)
                system_filename = f"document_{timestamp}_{len(results)}{ext}"
                
                # 保存文件
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], system_filename)
                file.save(temp_file_path)
                
                try:
                    # 读取文档内容
                    text = read_document(temp_file_path)
                    print(f"成功读取文档内容，长度: {len(text)} 字符")
                    
                    if not text.strip():
                        raise ValueError("文档内容为空")
                    
                    # 进行主题分析
                    topic_analysis = analyze_document_topics(text)
                    print("主题分析完成")
                    
                    # 准备文件信息
                    file_info = {
                        'filename': system_filename,
                        'original_filename': original_filename,
                        'original_text': text,
                        'mime_type': get_file_mime_type(original_filename),
                        'topic_analysis': topic_analysis
                    }
                    
                    # 生成摘要
                    summary = ollama_text(text, params, file_info, file.read())
                    
                    results.append({
                        'filename': original_filename,
                        'summary': summary,
                        'success': True
                    })
                    
                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as e:
                        print(f"清理临时文件失败: {str(e)}")
                        
            except Exception as e:
                print(f"处理文件 {file.filename} 时出错: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
                
        # 返回所有文件的处理结果
        return jsonify({
            'message': '文件处理完成',
            'results': results
        })
                
    except Exception as e:
        print(f"处理文档时发生错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'处理文档失败: {str(e)}'}), 500

def ollama_text(input_text, params=None, file_info=None, file_content=None):
    """使用Ollama生成文本摘要"""
    try:
        if not input_text:
            raise ValueError("输入文本不能为空")
            
        if not params:
            params = {}
            
        client = Client(host='http://localhost:11434')
        
        # 首先获取关键词
        print("正在生成关键词...")
        keyword_prompt = f"""请从以下文本中提取4个最重要的关键词,要求：
        1. 每个关键词2-4个字
        2. 用竖线分隔
        3. 直接输出关键词,不要其他内容
        4. 关键词应该反映文档的核心主题和内容
        
        文本内容：{input_text[:3000]}"""  # 限制文本长度为3000字符
        
        try:
            keyword_response = client.generate(
                model='huihui_ai/qwen2.5-1m-abliterated:latest',
                prompt=keyword_prompt,
                stream=False,
                options={'temperature': 0.4}
            )
            
            if not keyword_response or 'response' not in keyword_response:
                raise Exception("关键词API响应为空或格式错误")
                
            keywords = keyword_response['response'].strip()
            # 清理关键词文本，只保留实际的关键词
            keywords = re.sub(r'[^\w\u4e00-\u9fff|]', '', keywords)  # 只保留中文、字母、数字和分隔符
            keyword_list = keywords.split('|')
            
        except Exception as e:
            print(f"生成关键词失败: {str(e)}")
            print("使用默认关键词")
            keyword_list = ["文档摘要", "系统设计", "模型应用", "智能处理"]

        if len(keyword_list) != 4:
            print(f"警告：关键词数量不正确({len(keyword_list)})，进行调整")
            # 如果关键词不足4个，从文本中提取新的关键词补充
            if len(keyword_list) < 4:
                # 使用新的提示词尝试获取更多关键词
                additional_prompt = f"""从以下文本中再提取{4 - len(keyword_list)}个关键词，要求：
                1. 不要与已有关键词重复：{', '.join(keyword_list)}
                2. 每个关键词2-4个字
                3. 直接输出关键词，用竖线分隔
                
                文本内容：{input_text[:2000]}"""
                
                try:
                    additional_response = client.generate(
                        model='huihui_ai/qwen2.5-1m-abliterated:latest',
                        prompt=additional_prompt,
                        stream=False,
                        options={'temperature': 0.4}
                    )
                    
                    if not additional_response or 'response' not in additional_response:
                        raise Exception("补充关键词API响应为空或格式错误")
                        
                    additional_keywords = additional_response['response'].strip()
                    additional_keywords = re.sub(r'[^\w\u4e00-\u9fff|]', '', additional_keywords)
                    additional_list = additional_keywords.split('|')
                    keyword_list.extend(additional_list[:4 - len(keyword_list)])
                    
                except Exception as e:
                    print(f"获取补充关键词失败: {str(e)}")
                    # 使用默认值补充
                    while len(keyword_list) < 4:
                        keyword_list.append(f"主题{len(keyword_list)+1}")
            
            # 如果关键词超过4个，只保留前4个
            keyword_list = keyword_list[:4]
            
        # 确保每个关键词不超过8个字符
        keyword_list = [k[:8] for k in keyword_list]
        keywords = '|'.join(keyword_list)
        print(f"最终关键词: {keywords}")
        
        # 获取参数
        summary_length = params.get('summary_length', 'medium')
        target_language = params.get('target_language', 'chinese')
        summary_style = params.get('summary_style', '')
        focus_area = params.get('focus_area', '')
        expertise_level = params.get('expertise_level', '')
        language_style = params.get('language_style', '')
        output_format = params.get('output_format', 'paragraph')  # 新增输出格式参数
        
        # 更新字数映射为整数值
        summary_length_map = {
            'very_short': 200,
            'medium': 500,
            'long': 2000,
            'very_long': 5000
        }
        
        target_word_count = summary_length_map.get(summary_length, 500)

        # 风格说明映射
        style_descriptions = {
            'casual': '使用通俗易懂的语言，避免专业术语，适合一般读者',
            'academic': '使用学术性语言，包含专业术语和引用，适合学术读者',
            'professional': '使用专业分析的语言风格，包含行业术语，适合专业人士',
            'creative': '使用创意性的表达方式，生动有趣',
            'journalistic': '使用新闻报道的风格，客观中立，事实为主',
            'technical': '使用技术文档的风格，详细准确，步骤清晰',
            'business': '使用商务简报的风格，简洁直接，重点突出',
            'educational': '使用教育讲解的风格，通俗易懂，循序渐进'
        }
        
        # 输出格式映射
        format_descriptions = {
            'paragraph': '使用连续段落的形式组织内容',
            'bullet': '使用要点列表的形式组织内容，每个要点以"•"开始',
            'outline': '使用大纲结构组织内容，包含标题和子标题',
            'qa': '使用问答形式组织内容，针对文档的关键问题提供答案',
            'mindmap': '使用思维导图结构组织内容，从核心概念展开到各个方面'
        }
        
        # 专业程度映射
        expertise_descriptions = {
            'beginner': '面向入门级读者，解释基本概念，避免深入技术细节',
            'intermediate': '面向进阶读者，介绍适当的技术细节和背景知识',
            'advanced': '面向专家级读者，包含深入的技术讨论和专业分析',
            'expert': '面向权威级读者，包含最前沿的讨论和深度专业见解'
        }

        # 构建思维链提示词，包含所有前端参数
        summary_prompt = f"""你是一个专业的文档摘要专家。请按照思维链方法分析文档并生成高质量摘要。

[思维链步骤]
1. 分析：仔细阅读文档，确定主题、目的和主要论点
2. 提取：识别核心概念、关键信息点和重要论述
3. 判断类型：确定文档是学术论文、研究报告、技术文档还是一般文章
4. 分类整理：按照合适的结构组织内容
5. 提炼：从每个部分提取最具代表性的内容
6. 关键词确认：验证已提取的关键词是否准确反映文档核心内容
7. 整合：按照用户指定的参数要求生成最终摘要

[输出格式要求]
1. 首先输出 [KEYWORDS] 标记
2. 在其下方输出4个关键词，用竖线(|)分隔
3. 然后输出 [SUMMARY] 标记
4. 最后按照用户指定的格式输出摘要正文

[关键词]
{keywords}

[用户指定参数]
摘要长度：{target_word_count}字（±5%）
目标语言：{target_language}
摘要风格：{summary_style}（{style_descriptions.get(summary_style, '通用风格')}）
输出格式：{output_format}（{format_descriptions.get(output_format, '连续段落')}）
关注点：{focus_area}
专业程度：{expertise_level}（{expertise_descriptions.get(expertise_level, '适合一般读者')}）
语言风格：{language_style}

[摘要结构要求]
根据输出格式"{output_format}"和摘要风格"{summary_style}"，遵循以下结构：

1. 引言部分（约15%）：
   - 概述文档主题和背景
   - 点明核心问题或目的
   
2. 主体部分（约60%）：
   - 按照"{output_format}"格式组织文档的主要观点
   - 每个关键点需包含至少一个具体案例或数据支持
   - 使用"{expertise_level}"级别的专业术语和解释深度
   
3. 结论部分（约25%）：
   - 总结文档的主要发现或结论
   - 如果适用，提供实践建议或未来展望

[内容质量要求]
1. 确保摘要总长度为{target_word_count}字（±5%）
2. 使用"{language_style}"的语言风格
3. 重点关注"{focus_area}"方面的内容
4. 保持客观性和准确性，不添加原文中不存在的内容
5. 适当引用原文中的关键数据和证据支持观点

[原文内容]
{input_text}

请按照以上步骤和要求生成摘要，确保摘要的长度、风格、格式和内容符合用户指定的所有参数。
"""
        
        print("正在生成摘要...")
        
        def generate_with_retry(max_retries=3):
            """带重试的摘要生成函数"""
            original_target = target_word_count
            current_target = original_target
            retry_multipliers = [1.2, 1.5, 2.0]  # 分阶段调整系数
            
            for attempt in range(max_retries):
                try:
                    # 计算当前token数时增加冗余
                    current_num_predict = int(current_target * 2.5)  # 中文1 token ≈ 1.5字符
                    if attempt > 0:
                        current_num_predict = int(current_num_predict * retry_multipliers[attempt-1])
                        
                    # 在提示词中增加严格约束
                    current_prompt = summary_prompt.replace(
                        f"摘要长度：{target_word_count}字（±5%）",
                        f"""摘要长度：{current_target}字
                        1. 必须严格达到或略微超过{current_target}字
                        2. 如果未达字数要求，必须重新生成完整内容
                        3. 如果超过目标字数20%，需要适当精简"""
                    )
                    
                    response = client.generate(
                        model='huihui_ai/qwen2.5-1m-abliterated:latest',
                        prompt=current_prompt,
                        stream=False,
                        options={
                            'num_predict': min(current_num_predict, 16000),  # 不超过模型最大限制
                            'temperature': 0.7 + (0.1 * attempt),  # 逐步提高创造性
                            'top_p': 0.9,
                            'num_ctx': 16384,  # 确保足够上下文窗口
                            'stop': None
                        }
                    )
                    
                    if not response or 'response' not in response:
                        raise Exception("摘要API响应为空或格式错误")
                        
                    summary_text = response['response'].strip()
                    current_length = len(summary_text)
                    # 更严格的长度校验
                    min_length = int(original_target * 0.98)  # 允许2%误差
                    max_length = int(original_target * 1.2)  # 允许20%冗余
                    
                    if current_length >= min_length:
                        # 如果超过目标长度，截取到目标长度的2倍以内
                        if current_length > original_target * 2:
                            summary_text = summary_text[:original_target * 2]
                        print(f"生成摘要完成，长度: {len(summary_text)}字")
                        return summary_text
                    else:
                        print(f"警告：生成的摘要长度({current_length})不在预期范围内({min_length}-{max_length})，尝试重新生成")
                        if attempt < max_retries - 1:
                            print(f"增加目标字数到: {int(current_target * retry_multipliers[attempt])}")
                            continue
                        else:
                            print("达到最大重试次数，执行补偿机制")
                            return summary_text
                            
                except Exception as e:
                    print(f"第{attempt + 1}次生成摘要失败: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    raise
                    
        summary_text = generate_with_retry()
        
        # 最终长度补偿机制
        if len(summary_text) < target_word_count:
            print(f"执行最终补偿（当前{len(summary_text)}/目标{target_word_count}）")
            compensation_prompt = f"""请将以下摘要扩展到{target_word_count}字：
            
            [原始摘要]
            {summary_text}
            
            [扩展要求]
            1. 为每个主要观点添加具体案例
            2. 补充技术细节说明
            3. 增加数据支撑（可合理估算）
            4. 保持原有结构和逻辑
            5. 确保扩展后的内容连贯流畅
            
            请直接输出扩展后的完整摘要。"""
            
            try:
                compensation_response = client.generate(
                    model='huihui_ai/qwen2.5-1m-abliterated:latest',
                    prompt=compensation_prompt,
                    stream=False,
                    options={
                        'temperature': 0.8,
                            'top_p': 0.9,
                        'num_predict': target_word_count * 3,
                        'num_ctx': 16384,
                            'stop': None
                        }
                )
                
                if compensation_response and 'response' in compensation_response:
                    expanded_text = compensation_response['response'].strip()
                    if len(expanded_text) > len(summary_text):
                        summary_text = expanded_text
                        print(f"补偿成功，最终长度: {len(summary_text)}字")
                    
            except Exception as e:
                print(f"补偿机制执行失败: {str(e)}")

        # 更新文件信息
        if file_info is not None:
            file_info['keywords'] = keywords
        
        # 保存到数据库
        if file_info:
            try:
                save_summary_to_db(file_info, summary_text, params, file_content)
            except Exception as e:
                print(f"保存摘要到数据库失败: {str(e)}")
        
        return summary_text

    except Exception as e:
        print(f"生成摘要时发生错误: {str(e)}")
        traceback.print_exc()
        raise e

def init_admin():
    """初始化管理员账户"""
    try:
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                password=generate_password_hash('admin', method='pbkdf2:sha256'),  # 使用正确的哈希方法
                role='admin',
                created_at=datetime.now()
            )
            db.session.add(admin)
            db.session.commit()
            print("管理员账户初始化成功")
        else:
            # 更新管理员密码以确保使用正确的哈希方法
            admin.password = generate_password_hash('admin', method='pbkdf2:sha256')
            db.session.commit()
            print("管理员账户已存在，已更新密码哈希")
    except Exception as e:
        print(f"初始化管理员账户失败: {str(e)}")
        db.session.rollback()

@app.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400
            
        user = User.query.filter_by(username=username).first()
        
        if not user:
            return jsonify({'error': '用户名或密码错误'}), 401
            
        if check_password_hash(user.password, password):
            # 登录成功
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            
            return jsonify({
                'message': '登录成功',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            })
        else:
            return jsonify({'error': '用户名密码错误'}), 401
            
    except Exception as e:
        print(f"登录失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    """用户登出"""
    session.clear()
    return jsonify({'message': '登出成功'})

@app.route('/api/check_auth')
def check_auth():
    """检查用户认证状态"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session['username'],
                'role': session['role']
            }
        })
    return jsonify({'authenticated': False}), 401

@app.route('/')
def index():
    """主页"""
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/summary_library')
def summary_library():
    """染摘要库页面"""
    return render_template('summaries.html')

@app.route('/summaries')
def get_summaries():
    """获取所有摘要列表（分页）"""
    try:
        print("\n=== 开始获取摘要列表 ===")
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # 限制每页数量
        if per_page > 50:
            per_page = 50
            
        # 查询总数
        total = DocumentSummary.query.count()
        
        # 分页查询
        pagination = DocumentSummary.query.order_by(
            DocumentSummary.created_at.desc()
        ).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        summaries = pagination.items
        print(f"查询到 {len(summaries)} 条摘要记录")
        
        results = []
        for summary in summaries:
            # 使用原始文件名作为显示名称
            display_name = summary.original_filename or summary.display_filename or summary.file_name
            
            # 获取文件扩展名
            file_type = os.path.splitext(display_name)[1].lower().lstrip('.') if display_name else 'unknown'
            
            # 检查文件是否存在
            has_file = bool(summary.file_content) or (hasattr(summary, 'is_chunked') and summary.is_chunked)
            
            # 构建结果数据
            result_data = {
                'id': summary.id,
                'file_name': display_name,
                'file_type': file_type,
                'summary_text': summary.summary_text,
                'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'summary_length': summary.summary_length,
                'target_language': summary.target_language,
                'file_size': summary.file_size,
                'mime_type': summary.mime_type,
                'has_file': has_file,
                'is_chunked': getattr(summary, 'is_chunked', False),
                'total_chunks': getattr(summary, 'total_chunks', 0),
                'keywords': summary.keywords.split('|') if summary.keywords else []
            }
            
            results.append(result_data)
            
        # 构建分页信息
        pagination_info = {
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages,
            'has_prev': pagination.has_prev,
            'has_next': pagination.has_next,
            'prev_num': pagination.prev_num if pagination.has_prev else None,
            'next_num': pagination.next_num if pagination.has_next else None
        }
            
        print(f"\n返回 {len(results)} 条摘要记录")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"获取摘要列表错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/summaries/<int:summary_id>', methods=['GET'])
def get_summary_detail(summary_id):
    """获取单摘要详情"""
    try:
        print(f"\n=== 获取摘要详情 ID: {summary_id} ===")
        summary = DocumentSummary.query.get(summary_id)
        
        if not summary:
            print(f"未找到ID为 {summary_id} 的摘要")
            return jsonify({'error': f'未找到ID为 {summary_id} 的摘要'}), 404
            
        result = {
            'id': summary.id,
            'file_name': summary.file_name,
            'summary_text': summary.summary_text,
            'original_text': summary.original_text,
            'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': summary.summary_length,
            'target_language': summary.target_language
        }
        return jsonify(result)
        
    except Exception as e:
        print(f"获取摘要详情错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summaries/<int:summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    """删除指定摘要"""
    try:
        print(f"\n=== 开始删除摘要 ID: {summary_id} ===")
        
        # 首先查找摘要记录
        summary = DocumentSummary.query.get(summary_id)
        if not summary:
            error_msg = f"未找到ID为 {summary_id} 的摘要记录"
            print(error_msg)
            return jsonify({'error': error_msg}), 404
            
        print(f"找到摘要记录: {summary.file_name}")
        
        # 删除关联的文件映射记录
        mappings = FileMapping.query.filter_by(summary_id=summary_id).all()
        if mappings:
            print(f"删除 {len(mappings)} 个关联的文件映射记录")
            for mapping in mappings:
                db.session.delete(mapping)
        else:
            print("没有找到关联的文件映射记录")
        
        # 删除摘要记录
        print("删除摘要记录")
        db.session.delete(summary)
        
        # 提交事务
        db.session.commit()
        print(f"摘要删除成功")
        
        return jsonify({'message': '删除成功'})
            
    except Exception as e:
        print(f"删除过程中出错: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

def sanitize_filename(filename):
    """安全地处理文件名，保留中文字符"""
    # 移除文件名中的特殊字符，但保留中文字符
    name, ext = os.path.splitext(filename)
    # 只保留中文、英文、数字和下划线
    name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', name)
    # 确保文件名不为空
    if not name:
        name = 'document'
    return name + ext

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    extension = os.path.splitext(filename)[1].lower()
    print(f"检查文件扩展名: {extension}")
    print(f"允许的扩展名: {ALLOWED_EXTENSIONS}")
    # 移除扩展名前的点号再检查
    return extension.lstrip('.') in ALLOWED_EXTENSIONS

def analyze_document_topics(text):
    """分析文档主题"""
    try:
        print("=== 开始文档主题分析 ===")
        print(f"原始文本长度: {len(text)} 字符")
        
        # 调用 Ollama API 进行主题分析
        print("调用 Ollama API...")
        client = Client(host='http://localhost:11434')
        
        # 避免格式化错误，使用 repr() 移除可能导致格式字符串问题的特殊字符
        cleaned_text = repr(text[:2000]).strip("'")
        
        prompt = """请分析以下文本的主题，按照以下格式输出：
            1. 每个主题包含：标题、权重、关键词、描述
            2. 输出4个主题，权重总和为100%
            3. 每个主题包含3-5个关键词
            4. 每个主题提供简短描述
            5. 直接输出JSON格式数据
            6. 权重必须是数字类型，不能是字符串
            
            示例输出格式：
            {
                "topics": [
                    {
                        "title": "主题1",
                        "weight": 35.5,
                        "keywords": ["关键词1", "关键词2", "关键词3"],
                        "description": "这个主题主要讨论..."
                    },
                    // ... 其他主题
                ]
            }
            
            待分析文本：
            """ + cleaned_text
        
        response = client.generate(
            model='huihui_ai/qwen2.5-1m-abliterated:latest',
            prompt=prompt,
            stream=False,
            options={'temperature': 0.3}
        )
        
        if not response or 'response' not in response:
            raise Exception("API响应为空或格式错误")
            
        try:
            # 尝试解析JSON响应
            result = json.loads(response['response'].strip())
            if 'topics' not in result:
                raise ValueError("响应中缺少topics字段")
                
            # 验证和规范化主题数据
            topics = result['topics']
            if len(topics) != 4:
                print(f"警告：主题数量不正确({len(topics)})，将调整为4个主题")
                # 如果主题不足，添加默认主题
                while len(topics) < 4:
                    topics.append({
                        "title": f"主题{len(topics)+1}",
                        "weight": 0.0,
                        "keywords": [f"关键词{len(topics)+1}"],
                        "description": "自动生成的主题"
                    })
                # 如果主题过多，只保留前4个
                topics = topics[:4]
                
            # 确保权重总和为100%并且是数字类型
            for topic in topics:
                # 确保权重是浮点数
                if isinstance(topic['weight'], str):
                    try:
                        topic['weight'] = float(topic['weight'].replace('%', ''))
                    except (ValueError, TypeError):
                        topic['weight'] = 0.0
                elif not isinstance(topic['weight'], (int, float)):
                    topic['weight'] = 0.0
                
                # 处理标题和描述中可能包含的花括号，避免格式化错误
                if 'title' in topic and isinstance(topic['title'], str):
                    topic['title'] = re.sub(r'[{}]', '', topic['title'])
                
                if 'description' in topic and isinstance(topic['description'], str):
                    topic['description'] = re.sub(r'[{}]', '', topic['description'])
                
                # 处理关键词中可能包含的花括号
                if 'keywords' in topic and isinstance(topic['keywords'], list):
                    topic['keywords'] = [
                        re.sub(r'[{}]', '', k) if isinstance(k, str) else k
                        for k in topic['keywords']
                    ]
            
            # 计算总权重并归一化
            total_weight = sum(topic['weight'] for topic in topics)
            if total_weight == 0:
                # 如果总权重为0，平均分配
                for topic in topics:
                    topic['weight'] = 25.0
            elif total_weight != 100:
                print(f"警告：权重总和({total_weight}%)不等于100%，进行归一化")
                for topic in topics:
                    topic['weight'] = (topic['weight'] / total_weight) * 100.0
                    
            return {
                'success': True,
                'topics': topics
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {str(e)}")
            # 如果JSON解析失败，使用默认主题
            return {
                'success': True,
                'topics': [
                    {
                        "title": "技术创新",
                        "weight": 35.0,
                        "keywords": ["人工智能", "深度学习", "算法"],
                        "description": "涉及技术创新和发展"
                    },
                    {
                        "title": "应用实践",
                        "weight": 30.0,
                        "keywords": ["实施方案", "落地应用", "效果评估"],
                        "description": "关于技术的实际应用"
                    },
                    {
                        "title": "行业趋势",
                        "weight": 20.0,
                        "keywords": ["发展趋势", "市场分析", "前景展望"],
                        "description": "探讨行业发展方向"
                    },
                    {
                        "title": "挑战机遇",
                        "weight": 15.0,
                        "keywords": ["问题分析", "解决方案", "机遇把握"],
                        "description": "分析面临的挑战和机遇"
                    }
                ]
            }
            
    except Exception as e:
        print(f"主题分析失败: {str(e)}")
        # 返回默认主题，但标记为失败
        return {
            'success': False,
            'error': str(e),
            'topics': [
                {
                    "title": "技术创新",
                    "weight": 35.0,
                    "keywords": ["人工智能", "深度学习", "算法"],
                    "description": "涉及技术创新和发展"
                },
                {
                    "title": "应用实践",
                    "weight": 30.0,
                    "keywords": ["实施方案", "落地应用", "效果评估"],
                    "description": "关于技术的实际应用"
                },
                {
                    "title": "行业趋势",
                    "weight": 20.0,
                    "keywords": ["发展趋势", "市场分析", "前景展望"],
                    "description": "探讨行业发展方向"
                },
                {
                    "title": "挑战机遇",
                    "weight": 15.0,
                    "keywords": ["问题分析", "解决方案", "机遇把握"],
                    "description": "分析面临的挑战和机遇"
                }
            ]
        }

def create_hybrid_vector_store(text, summary, doc_id):
    """创建混合向量存储"""
    try:
        print("\n=== 开始创建混合向量存储 ===")
        
        # 参数检查
        if not isinstance(text, str):
            raise ValueError(f"原始文本必须是字符串类型，但实际类型是: {type(text)}")
        if not isinstance(summary, str):
            raise ValueError(f"摘要文本必须是字符串类型，但实际类型是: {type(summary)}")
        if not text.strip():
            print("警告: 原始文本为空，跳过向量存储创建")
            return
        if not summary.strip():
            print("警告: 摘要文本为空，跳过向量存储创建")
            return
            
        print(f"原始文本长度: {len(text)}")
        print(f"摘要文本长度: {len(summary)}")
        
        # 文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        
        # 分割文本
        text_chunks = text_splitter.split_text(text)
        summary_chunks = text_splitter.split_text(summary)
        
        print(f"正文分割为 {len(text_chunks)} 个块")
        print(f"摘要分割为 {len(summary_chunks)} 个块")
        
        # 使用 Ollama 嵌入
        embeddings = OllamaEmbeddings(
            model="huihui_ai/qwen2.5-1m-abliterated:latest",
            base_url="http://localhost:11434"
        )
        
        # 创建向量存储
        content_vectors = []
        summary_vectors = []
        
        # 处理正文块
        print("开始处理正文块...")
        for i, chunk in enumerate(text_chunks):
            try:
                vector = embeddings.embed_query(chunk)
                content_vectors.append({
                    'text': chunk,
                    'vector': vector,
                    'type': 'content',
                    'chunk_idx': i
                })
            except Exception as e:
                print(f"处理正文块 {i} 时出错: {str(e)}")
                continue
                
        # 处理摘要块
        print("开始处理摘要块...")
        for i, chunk in enumerate(summary_chunks):
            try:
                vector = embeddings.embed_query(chunk)
                summary_vectors.append({
                    'text': chunk,
                    'vector': vector,
                    'type': 'summary',
                    'chunk_idx': i
                })
            except Exception as e:
                print(f"处理摘要块 {i} 时出错: {str(e)}")
                continue
        
        print(f"成功处理 {len(content_vectors)} 个正文向量")
        print(f"成功处理 {len(summary_vectors)} 个摘要向量")
        
        # 将向量数据序列化并保存到数据库
        try:
            with app.app_context():
                summary = DocumentSummary.query.get(doc_id)
                if summary:
                    summary.content_vectors = pickle.dumps(content_vectors)
                    summary.summary_vectors = pickle.dumps(summary_vectors)
                    db.session.commit()
                    print("向量数据已成功保存到数据库")
                else:
                    print(f"未找到ID为 {doc_id} 的文档记录")
        except Exception as e:
            print(f"保存向量数据到数据库时出错: {str(e)}")
            raise
            
        print("=== 混合向量存储创建完成 ===\n")
        
    except Exception as e:
        print(f"创建向量存储失败: {str(e)}")
        print("继续处理,不中断流程")

@app.route('/analyze_topics/<int:summary_id>')
def analyze_topics(summary_id):
    """获取文档的主题分析"""
    try:
        print(f"\n=== 获取文档主题分析 ID: {summary_id} ===")
        summary = DocumentSummary.query.get_or_404(summary_id)
        
        def sanitize_topic_data(topic_data):
            """安全处理主题数据，确保数据格式正确"""
            try:
                if not isinstance(topic_data, dict):
                    # 如果不是字典类型，返回默认格式的主题分析数据
                    return {
                        'success': True,
                        'topics': [
                            {
                                "title": "技术创新",
                                "weight": 35.0,
                                "keywords": ["人工智能", "深度学习", "算法"],
                                "description": "涉及技术创新和发展"
                            },
                            {
                                "title": "应用实践",
                                "weight": 30.0,
                                "keywords": ["实施方案", "落地应用", "效果评估"],
                                "description": "关于技术的实际应用"
                            },
                            {
                                "title": "行业趋势",
                                "weight": 20.0,
                                "keywords": ["发展趋势", "市场分析", "前景展望"],
                                "description": "探讨行业发展方向"
                            },
                            {
                                "title": "挑战机遇",
                                "weight": 15.0,
                                "keywords": ["问题分析", "解决方案", "机遇把握"],
                                "description": "分析面临的挑战和机遇"
                            }
                        ]
                    }
                    
                sanitized_data = copy.deepcopy(topic_data)  # 使用深拷贝避免修改原始数据
                
                # 确保topics字段存在且是列表类型
                if 'topics' not in sanitized_data or not isinstance(sanitized_data['topics'], list):
                    sanitized_data['topics'] = []
                
                # 处理topics列表
                for i, topic in enumerate(sanitized_data.get('topics', [])):
                    if not isinstance(topic, dict):
                        # 如果主题不是字典类型，跳过处理
                        sanitized_data['topics'][i] = {
                            "title": f"主题{i+1}",
                            "weight": 25.0,
                            "keywords": [f"关键词{i+1}"],
                            "description": "自动生成的主题"
                        }
                        continue
                        
                    # 处理标题
                    if 'title' not in topic or not isinstance(topic['title'], str):
                        topic['title'] = f"主题{i+1}"
                    else:
                        # 移除可能导致格式问题的字符
                        topic['title'] = re.sub(r'[{}%]', '', topic['title'])
                        
                    # 处理描述
                    if 'description' not in topic or not isinstance(topic['description'], str):
                        topic['description'] = "自动生成的描述"
                    else:
                        # 移除可能导致格式问题的字符
                        topic['description'] = re.sub(r'[{}%]', '', topic['description'])
                        
                    # 处理关键词
                    if 'keywords' not in topic or not isinstance(topic['keywords'], list):
                        topic['keywords'] = [f"关键词{i+1}"]
                    else:
                        cleaned_keywords = []
                        for k in topic['keywords']:
                            if isinstance(k, str):
                                # 移除可能导致格式问题的字符
                                cleaned_keywords.append(re.sub(r'[{}%]', '', k))
                            else:
                                cleaned_keywords.append(str(k))
                        topic['keywords'] = cleaned_keywords
                        
                    # 确保权重是数字类型
                    if 'weight' not in topic:
                        topic['weight'] = 25.0  # 默认权重
                    else:
                        try:
                            if isinstance(topic['weight'], str):
                                # 清除百分号并转换为浮点数
                                weight_str = re.sub(r'[%{}]', '', topic['weight'])
                                # 处理空字符串的情况
                                if not weight_str.strip():
                                    topic['weight'] = 25.0
                                else:
                                    topic['weight'] = float(weight_str)
                            elif not isinstance(topic['weight'], (int, float)):
                                topic['weight'] = 25.0
                        except (ValueError, TypeError):
                            topic['weight'] = 25.0
                
                # 确保有4个主题
                while len(sanitized_data['topics']) < 4:
                    i = len(sanitized_data['topics'])
                    sanitized_data['topics'].append({
                        "title": f"主题{i+1}",
                        "weight": 25.0,
                        "keywords": [f"关键词{i+1}"],
                        "description": "自动生成的主题"
                    })
                
                # 如果主题超过4个，只保留前4个
                if len(sanitized_data['topics']) > 4:
                    sanitized_data['topics'] = sanitized_data['topics'][:4]
                    
                # 归一化权重总和为100%
                total_weight = sum(topic.get('weight', 0.0) for topic in sanitized_data['topics'])
                if total_weight <= 0:
                    # 如果总权重为0或负数，平均分配
                    for topic in sanitized_data['topics']:
                        topic['weight'] = 25.0
                else:
                    # 归一化权重
                    for topic in sanitized_data['topics']:
                        topic['weight'] = (topic['weight'] / total_weight) * 100.0
                
                # 确保success字段存在
                sanitized_data['success'] = True
                
                return sanitized_data
                
            except Exception as e:
                print(f"安全处理主题数据时出错: {str(e)}")
                # 返回默认主题数据
                return {
                    'success': True,
                    'topics': [
                        {
                            "title": "技术创新",
                            "weight": 35.0,
                            "keywords": ["人工智能", "深度学习", "算法"],
                            "description": "涉及技术创新和发展"
                        },
                        {
                            "title": "应用实践",
                            "weight": 30.0,
                            "keywords": ["实施方案", "落地应用", "效果评估"],
                            "description": "关于技术的实际应用"
                        },
                        {
                            "title": "行业趋势",
                            "weight": 20.0,
                            "keywords": ["发展趋势", "市场分析", "前景展望"],
                            "description": "探讨行业发展方向"
                        },
                        {
                            "title": "挑战机遇",
                            "weight": 15.0,
                            "keywords": ["问题分析", "解决方案", "机遇把握"],
                            "description": "分析面临的挑战和机遇"
                        }
                    ]
                }
            
        # 从数据库获取主题分析结果
        if summary.topic_analysis:
            print("从数据库获取已有的主题分析结果")
            if isinstance(summary.topic_analysis, dict) and 'topics' in summary.topic_analysis:
                # 安全处理数据后返回
                sanitized_result = sanitize_topic_data(summary.topic_analysis)
                return jsonify(sanitized_result)
            else:
                # 如果存储的格式不正确，重新生成
                print("存储的主题分析格式不正确，重新生成")
                topic_analysis = analyze_document_topics(summary.original_text)
                # 安全处理数据后存储和返回
                sanitized_analysis = sanitize_topic_data(topic_analysis)
                summary.topic_analysis = sanitized_analysis
                db.session.commit()
                return jsonify(sanitized_analysis)
        
        print("数据库中没有主题分析结果，开始新的分析")
        # 如果数据库中没有主题分析结果，进行分析
        topic_analysis = analyze_document_topics(summary.original_text)
        
        # 安全处理数据后存储和返回
        sanitized_analysis = sanitize_topic_data(topic_analysis)
        summary.topic_analysis = sanitized_analysis
        db.session.commit()
        
        return jsonify(sanitized_analysis)
        
    except Exception as e:
        print(f"获取主题分析失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'topics': []
        }), 500

@app.route('/preview/<int:summary_id>')
def preview_document(summary_id):
    """预览文档内容，支持分页加载"""
    try:
        print(f"\n=== 预览文档 ID: {summary_id} ===")
        summary = DocumentSummary.query.get_or_404(summary_id)
        print(f"找到文档: {summary.file_name}")
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 5, type=int)  # 默认每次加载5页
        print(f"当前页: {page}, 每页数量: {per_page}")
        
        # 获取文件类型
        file_type = os.path.splitext(summary.file_name)[1][1:].lower()
        print(f"文件类型: {file_type}")
        
        # 如果是PDF文件，使用PyMuPDF读取
        if file_type == 'pdf':
            # 从数据库获取文件内容
            file_content = get_file_content(summary.id)
            if not file_content:
                raise Exception("无法获取文件内容")
            print(f"获取到文件内容，大小: {len(file_content)} 字节")
            
            # 创建临时文件
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{summary.id}.pdf')
            try:
                with open(temp_file, 'wb') as f:
                    f.write(file_content)
                print(f"临时文件已创建: {temp_file}")
                
                # 使用PyMuPDF读取PDF
                doc = fitz.open(temp_file)
                total_pages = len(doc)
                print(f"PDF总页数: {total_pages}")
                
                # 计算当前批次要读取的页面范围
                start_page = (page - 1) * per_page
                end_page = min(start_page + per_page, total_pages)
                print(f"读取页面范围: {start_page + 1} - {end_page}")
                
                # 读取指定范围的页面
                current_content = []
                for page_num in range(start_page, end_page):
                    try:
                        print(f"正在读取第 {page_num + 1} 页...")
                        pdf_page = doc[page_num]
                        
                        # 获取页面尺寸
                        page_rect = pdf_page.rect
                        
                        # 提取文本，保持布局
                        page_text = pdf_page.get_text("text", sort=True)
                        
                        if page_text:
                            # 添加页码标记
                            page_content = f"=== 第 {page_num + 1} 页 ===\n{page_text}"
                            current_content.append(page_content)
                            print(f"第 {page_num + 1} 页内容长度: {len(page_text)}")
                        else:
                            print(f"第 {page_num + 1} 页未提取到内容")
                        
                    except Exception as e:
                        print(f"读取第 {page_num + 1} 页时出错: {str(e)}")
                        traceback.print_exc()
                        continue
                
                # 关闭文档
                doc.close()
                
                # 合并所有页面内容
                current_content = '\n\n'.join(current_content)
                print(f"合并后的内容长度: {len(current_content)}")
                
                # 如果是AJAX请求，返回JSON格式的分页内容
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    response_data = {
                        'content': current_content,
                        'has_more': end_page < total_pages,
                        'total_pages': total_pages,
                        'current_page': page
                    }
                    return jsonify(response_data)
                
                # 首次访问返回完整的预览页面
                print("返回完整预览页面")
                return render_template('preview.html', 
                    summary=summary,
                    initial_content=current_content,
                    total_pages=total_pages,
                    file_type=file_type
                )
            finally:
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"临时文件已删除: {temp_file}")
        else:
            # 非PDF文件使用原来的处理方式
            text_lines = summary.original_text.split('\n')
            lines_per_page = 50  # 每页显示50行
            total_pages = (len(text_lines) + lines_per_page - 1) // lines_per_page
            
            # 计算当前页的内容
            start_idx = (page - 1) * lines_per_page * per_page
            end_idx = min(start_idx + (lines_per_page * per_page), len(text_lines))
            current_content = '\n'.join(text_lines[start_idx:end_idx])
            
            # 如果是AJAX请求，返回JSON格式的分页内容
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'content': current_content,
                    'has_more': end_idx < len(text_lines),
                    'total_pages': total_pages,
                    'current_page': page
                })
            
            # 首次访问返回完整的预览页面
            return render_template('preview.html', 
                summary=summary,
                initial_content=current_content,
                total_pages=total_pages,
                file_type=file_type
            )
        
    except Exception as e:
        print(f"预览文档失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_embeddings_model():
    """获取嵌入模型"""
    try:
        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="snowflake-arctic-embed2"
        )
        return embeddings
    except Exception as e:
        print(f"加载嵌入模型失败: {str(e)}")
        traceback.print_exc()
        raise

def generate_semantic_summary(doc_id, query=None):
    """生成基于语义检索的摘要"""
    try:
        print(f"\n=== 生成语义摘要 文档ID: {doc_id} ===")
        
        # 如果没有提供查询，使用默认查询
        if not query:
            query = "总结这篇文档的主要内容和关键点"
            
        # 获取相关内容
        relevant_chunks = semantic_search(query, doc_id)
        
        # 构建上下文
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # 使用Ollama生成摘要
        llm = Ollama(model="huihui_ai/qwen2.5-1m-abliterated:latest")
        
        # 构建提示词
        prompt = f"""基于以下内容生成一个全面的摘要：

        {context}

        要求：
        1. 摘要应该清晰、连贯
        2. 突出重要信息和关键观点
        3. 保持客观性
        4. 控制在500字左右
        
        请直接输出摘要内容，不要包含任何额外说明。
        """
        
        # 生成摘要
        summary = llm(prompt)
        
        return summary.strip()
        
    except Exception as e:
        print(f"生成语义摘要失败: {str(e)}")
        traceback.print_exc()
        raise

# 添加新的路由处理语义搜索
@app.route('/semantic_search/<int:doc_id>', methods=['POST'])
def handle_semantic_search(doc_id):
    """处理语义搜索请求"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': '搜索查询不能为空'}), 400
            
        results = semantic_search(query, doc_id)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/semantic_summary/<int:doc_id>', methods=['POST'])
def handle_semantic_summary(doc_id):
    """处理语义摘要请求"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        summary = generate_semantic_summary(doc_id, query)
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def hybrid_semantic_search(query, doc_id, content_weight=0.6, summary_weight=0.4):
    """混合语义搜索"""
    try:
        print(f"\n=== 执行混合语义搜索 文档ID: {doc_id} ===")
        
        # 获取文档摘要
        summary = db.session.get(DocumentSummary, doc_id)
        if not summary:
            raise ValueError(f"未找到ID为 {doc_id} 的文档")
            
        # 检查向量数据是否存在
        if not summary.content_vectors or not summary.summary_vectors:
            print("向量数据不存在，创建新的向量存储")
            create_hybrid_vector_store(
                summary.original_text, 
                summary.summary_text, 
                doc_id
            )
            # 重新获取更新后的记录
            summary = db.session.get(DocumentSummary, doc_id)
            if not summary.content_vectors or not summary.summary_vectors:
                print("无法创建向量数据")
                return []
        else:
            try:
                # 从数据库加载向量数据
                content_vectors = pickle.loads(summary.content_vectors)
                summary_vectors = pickle.loads(summary.summary_vectors)
                print("成功从数据库加载向量数据")
            except Exception as e:
                print(f"加载向量数据失败: {str(e)}")
                content_vectors, summary_vectors = [], []
        
        if not content_vectors and not summary_vectors:
            print("没有可用的向量数据")
            return []
            
        # 获取嵌入模型
        embeddings = get_embeddings_model()
        
        # 生成查询向量
        query_vector = embeddings.embed_query(query)
        
        # 计算相似度并排序
        results = []
        
        # 处理正文向量
        for item in content_vectors:
            try:
                similarity = cosine_similarity(
                    [query_vector],
                    [item['vector']]
                )[0][0]
                results.append({
                    'content': item['text'],
                    'score': (1 - similarity) * content_weight,
                    'type': 'content',
                    'metadata': {
                        'type': item['type'],
                        'chunk_idx': item['chunk_idx']
                    }
                })
            except Exception as e:
                print(f"计算正文向量相似度时出错: {str(e)}")
                continue
        
        # 处理摘要向量
        for item in summary_vectors:
            try:
                similarity = cosine_similarity(
                    [query_vector],
                    [item['vector']]
                )[0][0]
                results.append({
                    'content': item['text'],
                    'score': (1 - similarity) * summary_weight,
                    'type': 'summary',
                    'metadata': {
                        'type': item['type'],
                        'chunk_idx': item['chunk_idx']
                    }
                })
            except Exception as e:
                print(f"计算摘要向量相似度时出错: {str(e)}")
                continue
        
        # 按得分排序（得分越低越相关）
        results.sort(key=lambda x: x['score'])
        
        return results[:5]  # 返回最相关的5个结果
        
    except Exception as e:
        print(f"混合语义搜索失败: {str(e)}")
        traceback.print_exc()
        return []

def generate_hybrid_semantic_summary(doc_id, query=None):
    """生成基于混合语义检索的摘要"""
    try:
        print(f"\n=== 生成混合语义摘要 文档ID: {doc_id} ===")
        
        # 如果没有提供查询，使用默认查询
        if not query:
            query = "总结这篇文档的主要内容和关键点"
            
        # 获取相关内容
        relevant_chunks = hybrid_semantic_search(query, doc_id)
        
        # 构建上下文（同时使用正文和摘要的相关内容）
        context_parts = []
        
        # 添加摘要内容
        summary_chunks = [chunk for chunk in relevant_chunks if chunk['type'] == 'summary']
        if summary_chunks:
            context_parts.append("摘要相关内容：")
            context_parts.extend([chunk['content'] for chunk in summary_chunks])
        
        # 添加正文内容
        content_chunks = [chunk for chunk in relevant_chunks if chunk['type'] == 'content']
        if content_chunks:
            context_parts.append("\n\n原文相关内容：")
            context_parts.extend([chunk['content'] for chunk in content_chunks])
        
        context = "\n\n".join(context_parts)
        
        # 使用Ollama生成摘要
        llm = Ollama(model="huihui_ai/qwen2.5-1m-abliterated:latest")
        
        # 构建提示词
        prompt = f"""基于以下内容生成一个全面的摘要：

        {context}

        要求：
        1. 摘要应该清晰、连贯
        2. 突出重要信息和关键观点
        3. 保持客观性
        4. 控制在500字左右
        5. 优先使用摘要中的表述，必要时参考原文内容补充细节
        
        请直接输出摘要内容，不要包含任何额外说明。
        """
        
        # 生成摘要
        summary = llm(prompt)
        
        return summary.strip()
        
    except Exception as e:
        print(f"生成混合语义摘要失败: {str(e)}")
        traceback.print_exc()
        raise

# 修改API路由以使用混合检索
@app.route('/hybrid_search/<int:doc_id>', methods=['POST'])
def handle_hybrid_search(doc_id):
    """处理混合语义搜索请求"""
    try:
        data = request.get_json()
        query = data.get('query')
        content_weight = data.get('content_weight', 0.6)
        summary_weight = data.get('summary_weight', 0.4)
        
        if not query:
            return jsonify({'error': '搜索查询不能为空'}), 400
            
        results = hybrid_semantic_search(query, doc_id, content_weight, summary_weight)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/hybrid_summary/<int:doc_id>', methods=['POST'])
def handle_hybrid_summary(doc_id):
    """处理混合语义摘要请求"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        summary = generate_hybrid_semantic_summary(doc_id, query)
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """搜索文档"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
            
        query = data.get('query')
        if not query:
            return jsonify({'error': '搜索关键词不能为空'}), 400
            
        print(f"\n=== 执行文档语义搜索 关键词: {query} ===")
        
        # 获取嵌入模型
        embeddings = get_embeddings_model()
        
        # 生成查询向量
        query_vector = embeddings.embed_query(query)
        
        results = []
        summaries = DocumentSummary.query.all()
        
        for summary in summaries:
            try:
                # 检查是否存在向量数据
                if not summary.content_vectors or not summary.summary_vectors:
                    # 如果不存在，创建向量存储
                    print(f"为文档 {summary.id} 创建向量存储")
                    create_hybrid_vector_store(summary.original_text, summary.summary_text, summary.id)
                    continue
                
                # 从数据库加载向量数据
                content_vectors = pickle.loads(summary.content_vectors)
                summary_vectors = pickle.loads(summary.summary_vectors)
                
                # 计算相似度得分
                max_similarity = 0
                
                # 检查正文向量
                for item in content_vectors:
                    similarity = cosine_similarity(
                        [query_vector],
                        [item['vector']]
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                # 检查摘要向量
                for item in summary_vectors:
                    similarity = cosine_similarity(
                        [query_vector],
                        [item['vector']]
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                # 如果相似度超过阈值
                if max_similarity > 0.3:  # 可以调整阈值
                    results.append({
                        'id': summary.id,
                        'file_name': summary.original_filename or summary.display_filename or summary.file_name,
                        'summary_text': summary.summary_text,
                        'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'target_language': summary.target_language,
                        'summary_length': summary.summary_length,
                        'score': max_similarity * 10,  # 将得分转换为0-10的范围
                        'keywords': summary.keywords.split('|') if summary.keywords else [],
                        'topic_analysis': summary.topic_analysis
                    })
            
            except Exception as e:
                print(f"处理文档 {summary.id} 时出错: {str(e)}")
                continue
        
        # 按相关度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'success': True,
            'results': results[:10]  # 只返回最相关的10个结果
        })
        
    except Exception as e:
        print(f"搜索文档时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        # 只在表不存在时创建表
        db.create_all()
        # 初始化管理员账户
        init_admin()
        print("数据库初始化完成")
    app.run(debug=True, port=5000)
