import pymysql
pymysql.install_as_MySQLdb()

from flask import Flask, request, jsonify, Response, render_template, stream_with_context, send_file, make_response, session, send_from_directory, redirect
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
import nltk
from time import sleep
import tempfile
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from chromadb import PersistentClient
from utils import process_response  # 导入process_response函数

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
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/doc_summary?charset=utf8mb4'
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary_length = db.Column(db.String(20))
    target_language = db.Column(db.String(20))
    file_size = db.Column(db.BigInteger)
    mime_type = db.Column(db.String(100))
    original_filename = db.Column(db.String(255))
    display_filename = db.Column(db.String(255))
    keywords = db.Column(db.String(255))
    topic_analysis = db.Column(db.JSON)
    embedding_model = db.Column(db.String(100))
    chunks_info = db.Column(db.JSON)
    is_chunked = db.Column(db.Boolean, default=False)
    total_chunks = db.Column(db.Integer, default=0)
    chroma_collection = db.Column(db.String(100))  # 新增：Chroma集合名称
    has_vector_store = db.Column(db.Boolean, default=False)  # 新增：是否已创建向量存储
    
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

class Summary(db.Model):
    """存储文本摘要记录"""
    __tablename__ = 'summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text(length=16777215), nullable=False)  # 原始文本
    summary_text = db.Column(db.Text(length=16777215), nullable=False)  # 摘要文本
    keywords = db.Column(db.String(255), nullable=True)  # 关键词
    model = db.Column(db.String(100), nullable=True)  # 使用的模型
    target_language = db.Column(db.String(20), nullable=True)  # 目标语言
    target_length = db.Column(db.String(20), nullable=True)  # 摘要长度
    focus_areas = db.Column(db.String(255), nullable=True)  # 关注领域
    level = db.Column(db.String(20), nullable=True)  # 专业程度
    language_style = db.Column(db.String(20), nullable=True)  # 语言风格
    style = db.Column(db.String(20), nullable=True)  # 写作风格
    format = db.Column(db.String(20), nullable=True)  # 输出格式
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # 创建时间
    
    __table_args__ = {
        'mysql_engine': 'InnoDB',
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }

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
        
        # 获取原始文本内容（如果有）
        original_text = file_info.get("original_text", "")
        
        # 计算文件内容的MD5哈希值
        import hashlib
        file_hash = hashlib.md5((original_text or "").encode('utf-8')).hexdigest()
        
        # 获取原始文件名和显示文件名
        original_filename = file_info.get("original_filename")
        display_filename = original_filename
        
        print(f"原始文件名: {original_filename}")
        print(f"显示文件名: {display_filename}")
        print(f"原始文本长度: {len(original_text) if original_text else 0}")

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
                
                # 保存原始文本内容（如果当前没有但新提供了）
                if not existing_summary.original_text and original_text:
                    existing_summary.original_text = original_text
                    print(f"更新现有摘要的原始文本内容")
                
                try:
                    db.session.commit()
                except Exception as e:
                    print(f"提交摘要更新时出错: {str(e)}")
                    db.session.rollback()
                
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
        
        # 检查文件是否存在且是否为有效的PDF
        if not os.path.exists(file_path):
            print(f"PDF文件不存在: {file_path}")
            return None
            
        # 检查文件大小，避免处理空文件
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"PDF文件为空: {file_path}")
            return None
            
        print(f"PDF文件大小: {file_size} 字节")
        
        # 首先尝试使用 PyMuPDF (更稳定的选择)
        try:
            with fitz.open(file_path) as doc:
                num_pages = doc.page_count
                print(f"PDF文件共有 {num_pages} 页 (PyMuPDF)")
                
                # 如果页数为0，可能不是有效的PDF
                if num_pages == 0:
                    print(f"PDF文件没有页面: {file_path}")
                    return None
                
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
                        else:
                            print(f"第 {page_num + 1} 页没有可提取的文本")
                        return page_num, ""
                    except Exception as e:
                        print(f"处理PDF第 {page_num + 1} 页时出错: {str(e)}")
                        return page_num, ""
                
                # 使用最多8个线程并行处理页面
                with ThreadPoolExecutor(max_workers=min(8, num_pages)) as executor:
                    results = list(executor.map(process_page_fitz, range(num_pages)))
                
                # 整理结果
                for page_num, page_text in results:
                    page_texts[page_num] = page_text
                
                # 合并所有页面文本
                text = "".join(page_texts)
                
                if not text.strip():
                    print(f"PDF文件未提取到文本内容: {file_path}")
                    # 可能是扫描件，需要OCR，但目前不处理
                
                print(f"成功从PDF提取文本，共 {len(text)} 个字符")
                return text
                
        except Exception as e:
            print(f"使用PyMuPDF读取PDF失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 继续尝试备用方法
        
        # 如果PyMuPDF失败，可以在这里添加备用PDF读取方法
        # 例如使用pdfplumber或PyPDF2等
        
        if not text.strip():
            print(f"未能从PDF提取文本: {file_path}")
            return None
            
        return text
            
    except Exception as e:
        print(f"PDF读取过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
    """读取文档内容，支持多种文档格式"""
    print(f"正在读取文档: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    print(f"文件扩展名: {file_extension}")
    
    # 基于扩展名决定使用哪个函数处理文件
    try:
        if file_extension == '.pdf':
            text = read_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:  # 同时支持 .docx 和 .doc
            text = read_docx(file_path)
        elif file_extension == '.txt':
            text = read_txt(file_path)
        elif file_extension == '.md':
            text = read_md(file_path)
        elif file_extension == '.epub':
            text = read_epub(file_path)
        else:
            # 尝试以文本方式读取
            try:
                print(f"未知扩展名: {file_extension}，尝试以文本方式打开")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if text and len(text.strip()) > 0:
                    print("以文本方式成功读取")
                    return text
            except Exception as e:
                print(f"以文本方式读取失败: {str(e)}")
            
            # 如果以文本方式读取失败，则报错
            print(f"不支持的文件格式: {file_extension}")
            raise ValueError(f"不支持的文件格式: {file_extension}")
        
        # 检查读取到的文本
        if not text or len(text.strip()) == 0:
            print(f"文件内容为空: {file_path}")
            return None
            
        print(f"成功读取文件，内容长度: {len(text)} 字符")
        return text
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
        summary = DocumentSummary.query.get_or_404(summary_id)
        
        if not summary:
            print(f"未找到ID为 {summary_id} 的摘要")
            return jsonify({'error': f'未找到ID为 {summary_id} 的摘要'}), 404
            
        # 使用原始文件名或显示文件名
        display_name = summary.original_filename or summary.display_filename or summary.file_name
        
        result = {
            'id': summary.id,
            'file_name': display_name,  # 使用正确的文件名
            'summary_text': summary.summary_text,
            'original_text': summary.original_text,
            'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': summary.summary_length,
            'target_language': summary.target_language,
            'keywords': summary.keywords.split('|') if summary.keywords else []  # 添加关键词信息
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
        
        # 删除关联的Chroma向量存储
        if summary.has_vector_store and summary.chroma_collection:
            try:
                print(f"尝试删除向量存储: {summary.chroma_collection}")
                collection_name = summary.chroma_collection
                persist_directory = "chroma_db"
                
                # 导入必要的库
                from langchain_chroma import Chroma
                from chromadb import PersistentClient
                
                # 方法1: 通过PersistentClient删除集合
                try:
                    print("方法1: 使用PersistentClient删除集合")
                    chroma_client = PersistentClient(path=persist_directory)
                    
                    # 兼容ChromaDB不同版本的API
                    try:
                        # ChromaDB v0.6.0+版本，list_collections直接返回集合名称列表
                        all_collections = set(chroma_client.list_collections())
                        print(f"ChromaDB v0.6.0+: 获取到 {len(all_collections)} 个集合名称")
                        if collection_name in all_collections:
                            chroma_client.delete_collection(collection_name)
                            print(f"成功通过PersistentClient删除集合: {collection_name}")
                        else:
                            print(f"集合不存在: {collection_name}")
                    except Exception:
                        # 旧版本API，集合对象有name属性
                        collections_list = chroma_client.list_collections()
                        collection_names = [col.name for col in collections_list]
                        if collection_name in collection_names:
                            chroma_client.delete_collection(collection_name)
                            print(f"成功通过PersistentClient删除集合: {collection_name}")
                        else:
                            print(f"集合不存在: {collection_name}")
                            
                except Exception as e:
                    print(f"通过PersistentClient删除集合失败: {str(e)}")
                    
                    # 方法2: 通过Chroma实例删除集合
                    try:
                        print("方法2: 使用Chroma实例删除集合")
                        from langchain_ollama import OllamaEmbeddings
                        
                        embeddings = OllamaEmbeddings(
                            model="snowflake-arctic-embed2",
                            base_url="http://localhost:11434"
                        )
                        
                        chroma_db = Chroma(
                            persist_directory=persist_directory,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        
                        if hasattr(chroma_db, 'delete_collection'):
                            chroma_db.delete_collection()
                            print(f"成功通过Chroma实例删除集合")
                        else:
                            print("Chroma实例没有delete_collection方法")
                            
                            # 备用方法3: 尝试访问底层客户端
                            if hasattr(chroma_db, '_client'):
                                if hasattr(chroma_db._client, 'delete_collection'):
                                    chroma_db._client.delete_collection(collection_name)
                                    print(f"成功通过_client删除集合")
                                else:
                                    print("chroma_db._client没有delete_collection方法")
                    except Exception as e2:
                        print(f"通过Chroma实例删除集合失败: {str(e2)}")
                
                print(f"向量存储删除处理完成")
            except Exception as e:
                print(f"删除向量存储时出错: {str(e)}")
                print("继续删除数据库记录")
        else:
            print("该摘要没有关联的向量存储或向量存储信息不完整")
        
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
        traceback.print_exc()
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
    """使用大模型分析文档主题，返回主题及其关键词"""
    try:
        client = Client(host='http://localhost:11434')
        
        # 限制输入文本长度，避免超出上下文窗口
        text_for_analysis = text[:8000] if len(text) > 8000 else text
        
        # 主题分析提示词
        topic_prompt = f"""请仔细分析以下文档，提取4-6个主要主题，并为每个主题提供相关关键词。

文档内容:
{text_for_analysis}

请以JSON格式返回分析结果，格式如下：
{{
  "topics": [
    {{
      "title": "主题名称",
      "weight": 35.0,
      "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
      "description": "对该主题的简要描述"
    }}
    // 更多主题...
  ]
}}

- 每个主题的权重(weight)之和应为100
- 确保每个主题有5-8个关键词
- 对每个主题提供简短的描述

仅返回JSON格式的结果，不要包含任何解释或其他文本。"""
        
        try:
            # 调用大模型进行主题分析
            topic_response = client.generate(
                model='huihui_ai/qwen2.5-1m-abliterated:latest',
                prompt=topic_prompt,
                stream=False,
                options={'temperature': 0.3}
            )
            
            if not topic_response or 'response' not in topic_response:
                raise Exception("主题分析API响应为空或格式错误")
            
            # 提取JSON响应
            response_text = topic_response['response'].strip()
            
            # 尝试解析JSON
            try:
                # 查找JSON对象的开始和结束位置
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # 确保包含必要的字段
                    if 'topics' not in result:
                        raise ValueError("响应中缺少topics字段")
                    
                    # 添加状态信息
                    result["success"] = True
                    
                    return result
                else:
                    raise ValueError("无法在响应中找到有效的JSON格式内容")
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                print(f"原始响应: {response_text}")
                raise Exception(f"无法解析主题分析结果: {str(e)}")
                
        except Exception as e:
            print(f"主题分析调用失败: {str(e)}")
            return get_default_topics()
            
    except Exception as e:
        print(f"主题分析错误: {str(e)}")
        return get_default_topics()

def get_default_topics():
    """返回默认的主题分析结果"""
    return {
        'success': True,
        'topics': [
            {
                "title": "主要内容",
                "weight": 35.0,
                "keywords": ["关键内容", "核心要点", "主要观点"],
                "description": "文档的主要内容和核心论述"
            },
            {
                "title": "技术方面",
                "weight": 30.0,
                "keywords": ["技术特点", "实现方式", "技术细节"],
                "description": "涉及的技术内容和实现方法"
            },
            {
                "title": "应用场景",
                "weight": 20.0,
                "keywords": ["使用场景", "应用领域", "实际应用"],
                "description": "文档描述的应用场景和使用方式"
            },
            {
                "title": "发展趋势",
                "weight": 15.0,
                "keywords": ["未来展望", "发展方向", "潜在影响"],
                "description": "相关领域的发展趋势和未来展望"
            }
        ]
    }

def get_embeddings_model():
    """获取统一的嵌入模型"""
    try:
        # 使用 snowflake-arctic-embed2 作为统一的嵌入模型
        embeddings = OllamaEmbeddings(
            model="snowflake-arctic-embed2",
            base_url="http://localhost:11434"
        )
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings model: {str(e)}")
        # 如果出现错误，仍然使用相同的模型重试，而不是切换到其他模型
        # 这样可以确保向量维度的一致性
        raise e

def create_hybrid_vector_store(text, summary, doc_id):
    """创建混合向量存储"""
    try:
        print("\n=== 开始创建混合向量存储 ===")
        print(f"文档ID: {doc_id}")
        print(f"原始文本长度: {len(text) if text else 0}")
        print(f"摘要文本长度: {len(summary) if summary else 0}")
        
        # 检查输入文本和摘要是否为空
        if not text or len(text.strip()) == 0:
            print("错误: 原始文本为空，无法创建向量存储")
            return False
            
        if not summary or len(summary.strip()) == 0:
            print("警告: 摘要文本为空，只创建原始文本的向量存储")
        
        print("第1步: 创建原始文本的向量存储")
        # 使用RAG工具创建向量存储
        # 为原文创建向量存储
        content_metadata = {
            'doc_id': doc_id,
            'source': 'content',
            'type': 'original'
        }
        try:
            print(f"正在调用RAGTools.create_vector_store为原始文本创建向量存储...")
            content_success = rag_tools.create_vector_store(text, doc_id, content_metadata)
            print(f"原始文本向量存储创建结果: {'成功' if content_success else '失败'}")
        except Exception as e:
            print(f"创建原始文本向量存储时出错: {str(e)}")
            content_success = False
        
        # 如果摘要不为空，则为摘要创建向量存储
        if summary and len(summary.strip()) > 0:
            print("第2步: 创建摘要文本的向量存储")
            summary_metadata = {
                'doc_id': doc_id,
                'source': 'summary',
                'type': 'summary'
            }
            try:
                print(f"正在调用RAGTools.create_vector_store为摘要文本创建向量存储...")
                summary_success = rag_tools.create_vector_store(summary, doc_id, summary_metadata)
                print(f"摘要文本向量存储创建结果: {'成功' if summary_success else '失败'}")
            except Exception as e:
                print(f"创建摘要文本向量存储时出错: {str(e)}")
                summary_success = False
        else:
            print("跳过第2步: 摘要文本为空")
            summary_success = True  # 如果摘要为空，我们认为这不是失败
        
        # 更新数据库记录，标记为已创建向量存储
        try:
            print("第3步: 更新数据库记录，标记文档已创建向量存储")
            doc = DocumentSummary.query.get(doc_id)
            if doc:
                doc.has_vector_store = True
                doc.chroma_collection = f"doc_{doc_id}"
                db.session.commit()
                print(f"成功更新数据库记录，文档ID {doc_id} 已标记为创建了向量存储")
            else:
                print(f"警告: 找不到文档ID {doc_id} 的记录，无法更新向量存储状态")
        except Exception as e:
            print(f"更新数据库记录时出错: {str(e)}")
            # 不中断流程，继续返回向量存储创建结果
        
        result = content_success and summary_success
        print(f"=== 混合向量存储创建{'成功' if result else '失败'} ===\n")
        return result
        
    except Exception as e:
        print(f"创建混合向量存储时出错: {str(e)}")
        traceback.print_exc()
        return False

def hybrid_semantic_search(query, doc_id, content_weight=0.6, summary_weight=0.4):
    """混合语义搜索"""
    try:
        print(f"\n=== 执行混合语义搜索 文档ID: {doc_id} ===")
        
        # 使用RAG工具进行语义搜索
        results = rag_tools.semantic_search(query, doc_id, top_k=8)
        
        if not results:
            return []
            
        # 根据来源应用权重
        for result in results:
            source = result['metadata'].get('source', 'content')
            result['score'] = result['score'] * (content_weight if source == 'content' else summary_weight)
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:8]
        
    except Exception as e:
        print(f"混合语义搜索时出错: {str(e)}")
        traceback.print_exc()
        return []

def generate_semantic_summary(doc_id, query=None):
    """生成基于语义检索的摘要"""
    try:
        print(f"\n=== 生成语义摘要 文档ID: {doc_id} ===")
        
        # 如果没有提供查询，使用默认查询
        if not query:
            query = "总结这篇文档的主要内容和关键点"
            
        # 获取相关内容
        relevant_chunks = semantic_search(query, doc_id)
        
        # 如果没有找到相关内容，返回提示信息
        if not relevant_chunks:
            return "未能找到与查询相关的内容，无法生成摘要。"
            
        # 构建上下文
        try:
            context = "\n\n".join([chunk.get('content', '') for chunk in relevant_chunks])
        except (AttributeError, KeyError) as e:
            print(f"构建上下文时出错: {str(e)}")
            # 尝试使用其他可能的字段名
            context = "\n\n".join([chunk.get('text', '') for chunk in relevant_chunks])
        
        # 使用Ollama生成摘要
        llm = Ollama(model="huihui_ai/qwen2.5-1m-abliterated:latest")
        
        # 构建提示词
        prompt = f"""请你是一个专业的文档摘要分析师。根据以下文档，生成一个{summary_length_text}，使用{target_language_text}，{style_text}。
{focus_text}，{level_text}，{lang_style_text}，{format_text}。

【重要长度要求】：必须严格生成长度为{target_word_count}字的摘要，不能少于此字数的90%。如果内容不足，请通过添加更多细节、解释和具体案例来达到要求字数。

【格式要求】：
- 首先输出标记[KEYWORDS]，然后在下一行列出4-6个关键词，以竖线(|)分隔
- 然后输出标记[SUMMARY]，之后开始你的摘要正文
- 摘要应当包含充分的解释、分析和支持性细节，以达到{target_word_count}字

请使用以下思维链步骤来生成高质量摘要：

步骤1：深入阅读文档，确定文档的主题、目的和主要观点。
步骤2：提取关键信息和中心思想，包括核心主题、主要论点、关键证据和结论。
步骤3：分析文档的结构和逻辑流程，确定各个部分之间的关系。
步骤4：构建一个连贯、完整的摘要框架，确保能支撑{target_word_count}字的详细内容。
步骤5：生成详细摘要，确保：
   - 每个关键点都有充分展开，提供足够的事实和数据支持
   - 对复杂概念进行深入解释和分析
   - 添加具体案例和应用场景说明
   - 提供必要的背景信息和上下文
   - 确保总字数达到{target_word_count}字

==== 文档内容 ====
{input_text}
==== 文档内容结束 ====

首先输出[KEYWORDS]和关键词，然后输出[SUMMARY]和摘要正文。务必确保摘要长度达到{target_word_count}字："""

        # 创建客户端
        print("开始调用大模型生成摘要")

        try:
            # 调用模型API - 流式响应
            response_stream = client.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={
                    'num_predict': num_predict_tokens,  # 使用前面计算的预测token数量
                    'temperature': 0.8,  # 适当提高温度以获得更多样化的输出
                    'top_p': 0.9,
                    'num_ctx': context_length,  # 使用前面计算的上下文长度
                    'stop': None
                }
            )
            
            # 初始状态变量
            buffer = ""
            keywords_section = ""
            summary_text = ""  # 用于收集完整的摘要文本
            keywords_started = False
            keywords_completed = False
            summary_started = False
            
            print("已开始流式响应")
            
            # 直接发送一个换行，确保前端开始显示
            yield "\n"
            
            for response_chunk in response_stream:
                if 'response' in response_chunk:
                    token = response_chunk['response']
                    buffer += token
                    
                    # 检测标记状态
                    if "[KEYWORDS]" in buffer and not keywords_started:
                        keywords_started = True
                        print("检测到关键词段开始")
                        continue
                    
                    if "[SUMMARY]" in buffer and not summary_started:
                        summary_started = True
                        keywords_completed = True
                        print("检测到摘要段开始")
                        # 摘要开始，发送间隔符
                        yield "\n\n"
                        continue
                    
                    # 当找到[KEYWORDS]后，将token添加到keywords_section
                    if keywords_started and not keywords_completed:
                        keywords_section += token
                        continue
                    
                    # 当找到[SUMMARY]后，直接流式输出每个token，并收集完整摘要
                    if summary_started:
                        # 过滤掉[SUMMARY]标记本身
                        if token not in "[SUMMARY]":
                            summary_text += token  # 收集完整摘要
                            yield token
            
            # 处理特殊情况：如果没有找到[SUMMARY]标记但已经结束
            if not summary_started:
                print("没有找到明确的[SUMMARY]标记")
                # 如果有关键词但没有摘要标记
                if keywords_started:
                    # 尝试在关键词后找到第一个换行作为摘要开始
                    if "\n" in keywords_section:
                        summary_text = keywords_section.split("\n", 1)[1].strip()
                        if summary_text:
                            print("使用关键词后的内容作为摘要")
                            yield "\n\n" + summary_text
                        else:
                            # 如果没有有效的摘要内容，则发送一个提示
                            summary_text = "无法从响应中提取摘要内容，请重试。"
                            yield "\n\n" + summary_text
                else:
                    # 没有关键词标记，将整个buffer作为摘要
                    print("使用完整响应作为摘要")
                    summary_text = buffer
                    yield buffer
        
        except Exception as e:
            error_message = f"生成摘要时发生错误: {str(e)}"
            print(error_message)
            yield "\n\n" + error_message

    except Exception as e:
        print(f"生成语义摘要失败: {str(e)}")
        traceback.print_exc()
        return "摘要生成失败，请稍后重试。"

# 添加新的路由处理语义搜索
@app.route('/semantic_search/<int:doc_id>', methods=['POST'])
def handle_semantic_search(doc_id):
    """处理语义搜索请求"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({
                'success': False, 
                'error': '搜索查询不能为空',
                'results': []
            }), 400
            
        results = semantic_search(query, doc_id)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"语义搜索处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'results': []
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
        print(f"语义摘要处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'summary': ""
        }), 500

def generate_hybrid_semantic_summary(doc_id, query=None):
    """生成基于混合语义检索的摘要"""
    try:
        print(f"\n=== 生成混合语义摘要 文档ID: {doc_id} ===")
        
        # 使用RAG工具生成响应
        response = rag_tools.generate_rag_response(query or "总结这篇文档的主要内容和关键点", doc_id)
        
        if not response:
            return "无法生成摘要，请稍后重试。"
            
        return response['answer']
        
    except Exception as e:
        print(f"生成混合语义摘要失败: {str(e)}")
        traceback.print_exc()
        return "摘要生成失败，请稍后重试。"

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
            return jsonify({
                'success': False,
                'error': '搜索查询不能为空',
                'results': []  # 即使出错也返回空结果数组
            }), 400
            
        results = hybrid_semantic_search(query, doc_id, content_weight, summary_weight)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"混合搜索处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'results': []  # 确保即使出错也返回空结果数组
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
        print(f"混合摘要处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'summary': ""  # 确保即使出错也返回空摘要字符串
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """搜索文档"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False, 
                'error': '请求数据不能为空',
                'results': []
            }), 400
            
        query = data.get('query')
        if not query:
            return jsonify({
                'success': False, 
                'error': '搜索关键词不能为空',
                'results': []
            }), 400
            
        print(f"\n=== 执行文档语义搜索 关键词: {query} ===")
        
        # 检查 Ollama 服务是否可用
        try:
            embeddings = get_embeddings_model()
            # 生成查询向量
            query_vector = embeddings.embed_query(query)
        except Exception as e:
            print(f"Ollama 服务不可用: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Ollama 服务不可用，请确保服务已启动并正常运行',
                'results': []
            }), 503
        
        results = []
        # 获取所有有向量数据的文档
        summaries = DocumentSummary.query.filter(
            db.and_(
                DocumentSummary.content_vectors.isnot(None),
                DocumentSummary.summary_vectors.isnot(None)
            )
        ).all()
        
        for summary in summaries:
            try:
                # 从数据库加载向量数据
                content_vectors = pickle.loads(summary.content_vectors)
                summary_vectors = pickle.loads(summary.summary_vectors)
                
                # 计算最大相似度
                max_similarity = 0
                best_match_text = ""
                
                # 检查正文向量
                for item in content_vectors:
                    try:
                        similarity = cosine_similarity(
                            [query_vector],
                            [item['vector']]
                        )[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match_text = item['text']
                    except Exception as e:
                        print(f"计算正文向量相似度时出错: {str(e)}")
                        continue
                
                # 检查摘要向量
                for item in summary_vectors:
                    try:
                        similarity = cosine_similarity(
                            [query_vector],
                            [item['vector']]
                        )[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match_text = item['text']
                    except Exception as e:
                        print(f"计算摘要向量相似度时出错: {str(e)}")
                        continue
                
                # 如果相似度超过阈值
                if max_similarity > 0.3:  # 可以调整阈值
                    # 使用原始文件名作为显示名称
                    display_name = summary.original_filename or summary.display_filename or summary.file_name
                    
                    results.append({
                        'id': summary.id,
                        'file_name': display_name,
                        'summary_text': summary.summary_text,
                        'best_match_text': best_match_text,  # 添加最佳匹配文本
                        'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'target_language': summary.target_language,
                        'summary_length': summary.summary_length,
                        'score': float(max_similarity),  # 保持原始相似度分数
                        'keywords': summary.keywords.split('|') if summary.keywords else [],
                        'topic_analysis': summary.topic_analysis
                    })
            
            except Exception as e:
                print(f"处理文档 {summary.id} 时出错: {str(e)}")
                continue
        
        # 按相关度排序（分数越高越相关）
        results.sort(key=lambda x: x['score'], reverse=True)
            
        return jsonify({
            'success': True,
            'results': results,
            'message': '未找到相关文档' if not results else None
        })
        
    except Exception as e:
        print(f"搜索文档时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'results': []
        }), 500

# 添加用户管理相关路由
@app.route('/admin/users')
def admin_users():
    """用户管理页面"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    return render_template('admin/users.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    """获取所有用户列表"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': '未授权访问'}), 401
    
    try:
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """获取单个用户信息"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': '未授权访问'}), 401
    
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    """创建新用户"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': '未授权访问'}), 401
    
    try:
        data = request.get_json()
        if not all(k in data for k in ['username', 'email', 'password', 'role']):
            return jsonify({'error': '缺少必要的字段'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': '用户名已存在'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': '邮箱已存在'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password=generate_password_hash(data['password']),
            role=data['role']
        )
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': '用户创建成功',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """更新用户信息"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': '未授权访问'}), 401
    
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        if 'username' in data and data['username'] != user.username:
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'error': '用户名已存在'}), 400
            user.username = data['username']
        
        if 'email' in data and data['email'] != user.email:
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'error': '邮箱已存在'}), 400
            user.email = data['email']
        
        if 'password' in data and data['password']:
            user.password = generate_password_hash(data['password'])
        
        if 'role' in data:
            user.role = data['role']
        
        db.session.commit()
        return jsonify({'message': '用户更新成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """删除用户"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': '未授权访问'}), 401
    
    try:
        user = User.query.get_or_404(user_id)
        if user.username == 'admin':
            return jsonify({'error': '不能删除管理员账户'}), 400
        
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': '用户删除成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# 添加主题分析API路由
@app.route('/analyze_topics/<int:summary_id>', methods=['GET'])
def analyze_topics(summary_id):
    """获取文档主题分析结果"""
    try:
        # 查询文档摘要记录
        doc = DocumentSummary.query.get(summary_id)
        if not doc:
            return jsonify({
                'success': False,
                'error': f'未找到ID为{summary_id}的文档'
            }), 404
            
        # 检查是否已有主题分析数据
        if doc.topic_analysis:
            # 如果已有数据，直接返回
            return jsonify(doc.topic_analysis)
        
        # 没有分析数据，使用文档内容进行分析
        if not doc.original_text:
            return jsonify({
                'success': False,
                'error': '文档内容为空，无法进行主题分析'
            }), 400
            
        # 执行主题分析
        analysis_result = analyze_document_topics(doc.original_text)
        
        # 保存分析结果到数据库
        doc.topic_analysis = analysis_result
        db.session.commit()
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"主题分析API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'主题分析失败: {str(e)}'
        }), 500

def semantic_search(query, doc_id):
    """语义搜索函数"""
    try:
        print(f"\n=== 执行语义搜索 文档ID: {doc_id} ===")
        
        # 获取文档
        doc = DocumentSummary.query.get(doc_id)
        if not doc:
            print(f"未找到文档ID: {doc_id}")
            return []
            
        # 检查向量数据
        if not doc.content_vectors:
            print(f"文档 {doc_id} 没有向量数据")
            return []
            
        # 获取统一的嵌入模型
        embeddings = get_embeddings_model()
        
        # 生成查询向量
        query_vector = embeddings.embed_query(query)
        
        # 从数据库加载向量数据
        content_vectors = pickle.loads(doc.content_vectors)
        
        # 计算相似度并存储结果
        results = []
        
        # 处理正文向量
        for item in content_vectors:
            try:
                similarity = cosine_similarity(
                    [query_vector],
                    [item['vector']]
                )[0][0]
                
                results.append({
                    "text": item['text'],
                    "content": item['text'],  # 为了兼容旧代码
                    "score": float(similarity),
                    "source": "content",
                    "metadata": {"index": item['index']}
                })
            except Exception as e:
                print(f"计算正文向量相似度时出错: {str(e)}")
                continue
        
        # 按分数排序（分数越高越相关）
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 只返回前8个最相关的结果
        return results[:8]
        
    except Exception as e:
        print(f"语义搜索时出错: {str(e)}")
        traceback.print_exc()
        return []

@app.route('/process_document_stream', methods=['POST'])
def process_document_stream():
    """处理文档并生成摘要 - 流式输出版本"""
    print("接收到流式处理请求")
    if 'file' not in request.files:
        return jsonify({'error': '请选择至少一个文件'}), 400

    files = request.files.getlist('file')
    print(f"接收到 {len(files)} 个文件")
    
    if not files:
        return jsonify({'error': '请选择至少一个文件'}), 400
    
    # 获取参数
    summary_length = request.form.get('summary_length', 'medium')
    target_language = request.form.get('target_language', 'chinese')
    summary_style = request.form.get('summary_style', 'basic')
    output_format = request.form.get('output_format', 'narrative')
    focus_area = request.form.get('focus_area', 'analytical')
    expertise_level = request.form.get('expertise_level', 'deductive')
    language_style = request.form.get('language_style', 'precise')
    
    # 转换为字典
    params = {
        'summary_length': summary_length,
        'target_language': target_language,
        'summary_style': summary_style,
        'output_format': output_format,
        'focus_area': focus_area, 
        'expertise_level': expertise_level,
        'language_style': language_style
    }
    print(f"处理参数: {params}")
    
    # 临时文件路径列表，用于在响应完成后清理
    temp_files = []
    
    def generate_stream():
        """生成流式响应的生成器函数"""
        nonlocal temp_files  # 使用nonlocal访问外部作用域的变量
        
        try:
            for idx, file in enumerate(files):
                if not file.filename:
                    print(f"跳过文件 {idx}，文件名为空")
                    continue

                try:
                    print(f"开始处理文件 {idx}: {file.filename}")
                    # 发送文件开始标记
                    yield f"FILE_START:{idx}:{file.filename}\n"
                    
                    # 读取文件内容到内存而不写入临时文件
                    file_content = file.read()
                    if not file_content:
                        yield f"FILE_ERROR:{idx}:文件内容为空\n"
                        continue
                        
                    # 提取文件扩展名
                    original_filename = file.filename
                    file_extension = ""
                    if '.' in original_filename:
                        file_extension = original_filename.rsplit('.', 1)[1].lower()
                    
                    # 获取文件信息
                    file_size = len(file_content)
                    mime_type = file.content_type if hasattr(file, 'content_type') else None
                    
                    # 如果没有MIME类型或不准确，尝试通过文件名推断
                    if not mime_type or mime_type == 'application/octet-stream':
                        mime_type = get_file_mime_type(original_filename)
                    
                    print(f"文件 {idx} 的信息: 原始文件名={original_filename}, 大小={file_size}, MIME类型={mime_type}")
                    
                    # 创建文件信息对象
                    file_info = {
                        'filename': f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.{file_extension}" if file_extension else f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}",
                        'original_filename': original_filename,
                        'size': file_size,
                        'mime_type': mime_type
                    }
                    
                    # 创建用于处理的文本 - 从二进制内容中提取
                    text = None
                    original_text = None
                    
                    # 尝试提取并保存原始文本内容
                    try:
                        print(f"尝试从文件内容提取原始文本: {original_filename}")
                        
                        if file_extension.lower() == 'pdf':
                            # 对于PDF文件，使用PyMuPDF提取文本
                            try:
                                # 创建临时文件用于PyMuPDF处理
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                                    temp_file.write(file_content)
                                    temp_path = temp_file.name
                                    temp_files.append(temp_path)  # 添加到临时文件列表

                                try:
                                    doc = fitz.open(temp_path)
                                    text_parts = []
                                    for page_num in range(len(doc)):
                                        text_parts.append(doc[page_num].get_text())
                                    original_text = "\n\n".join(text_parts)
                                    text = original_text  # 用于生成摘要
                                    doc.close()
                                    print(f"成功从PDF提取文本，长度: {len(original_text) if original_text else 0}")
                                finally:
                                    # 确保临时文件被删除
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                        temp_files.remove(temp_path)
                            except Exception as e:
                                print(f"PDF文本提取错误: {str(e)}")
                        elif file_extension.lower() in ['docx', 'doc']:
                            # 对于Word文档，使用python-docx提取文本
                            try:
                                # 创建临时文件用于python-docx处理
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                                    temp_file.write(file_content)
                                    temp_path = temp_file.name
                                    temp_files.append(temp_path)  # 添加到临时文件列表
                                    
                                try:
                                    doc = Document(temp_path)
                                    text_parts = []
                                    for para in doc.paragraphs:
                                        text_parts.append(para.text)
                                    original_text = "\n".join(text_parts)
                                    text = original_text  # 用于生成摘要
                                    print(f"成功从Word文档提取文本，长度: {len(original_text) if original_text else 0}")
                                finally:
                                    # 确保临时文件被删除
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                        temp_files.remove(temp_path)
                            except Exception as e:
                                print(f"Word文档文本提取错误: {str(e)}")
                        elif file_extension.lower() in ['txt', 'md']:
                            # 对于纯文本文件，直接解码
                            try:
                                try:
                                    original_text = file_content.decode('utf-8')
                                except UnicodeDecodeError:
                                    original_text = file_content.decode('latin-1')
                                text = original_text  # 用于生成摘要
                                print(f"成功从文本文件解码内容，长度: {len(original_text) if original_text else 0}")
                            except Exception as e:
                                print(f"文本文件解码错误: {str(e)}")
                        elif file_extension.lower() == 'epub':
                            # 对于EPUB文件，使用ebooklib提取文本
                            try:
                                # 创建临时文件用于ebooklib处理
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_file:
                                    temp_file.write(file_content)
                                    temp_path = temp_file.name
                                    temp_files.append(temp_path)  # 添加到临时文件列表
                                    
                                try:
                                    book = epub.read_epub(temp_path)
                                    text_parts = []
                                    for item in book.get_items():
                                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                                            soup = BeautifulSoup(item.get_content(), 'html.parser')
                                            text_parts.append(soup.get_text())
                                    original_text = "\n\n".join(text_parts)
                                    text = original_text  # 用于生成摘要
                                    print(f"成功从EPUB提取文本，长度: {len(original_text) if original_text else 0}")
                                finally:
                                    # 确保临时文件被删除
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                        temp_files.remove(temp_path)
                            except Exception as e:
                                print(f"EPUB文本提取错误: {str(e)}")
                        
                        # 如果成功提取了文本，更新file_info对象
                        if original_text:
                            file_info['original_text'] = original_text
                            print(f"已将提取的原始文本添加到file_info")
                    except Exception as e:
                        print(f"文本提取整体过程错误: {str(e)}")
                        # 这里捕获但不抛出异常，允许继续处理
                    
                    # 如果文本提取失败，尝试使用read_document函数
                    if not text:
                        # 需要使用临时文件以便read_document函数处理
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}" if file_extension else "") as temp_file:
                                temp_file.write(file_content)
                                temp_path = temp_file.name
                                temp_files.append(temp_path)  # 添加到临时文件列表
                            
                            try:
                                print(f"使用read_document函数读取文本")
                                text = read_document(temp_path)
                                
                                # 如果read_document成功提取了文本，但之前的方法失败了，更新file_info
                                if text and not original_text:
                                    file_info['original_text'] = text
                                    print(f"通过read_document成功提取文本，长度: {len(text)}")
                            except Exception as e:
                                print(f"使用read_document读取失败: {str(e)}")
                            finally:
                                # 确保临时文件被删除
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    if temp_path in temp_files:
                                        temp_files.remove(temp_path)
                        except Exception as e:
                            print(f"创建临时文件失败: {str(e)}")
                    
                    if not text:
                        yield f"FILE_ERROR:{idx}:无法提取文本内容\n"
                        continue
                
                    # 创建摘要生成请求的任务ID
                    task_id = f"stream_task_{uuid.uuid4()}"
                    print(f"任务ID: {task_id}")
                    
                    # 调用大模型生成摘要 - 使用流式输出版本
                    print(f"开始生成摘要，文本长度: {len(text)}")
                    token_count = 0
                    for token in ollama_text_stream(text, params, file_info, file_content):
                        token_count += 1
                        if token_count % 100 == 0:
                            print(f"已生成 {token_count} 个token")
                        yield token
                    
                    print(f"摘要生成完成，共 {token_count} 个token")
                    # 发送文件结束标记
                    yield f"FILE_END:{idx}\n"
                
                except Exception as e:
                    print(f"处理文件时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 发送错误信息
                    error_message = str(e).replace('\n', ' ')
                    yield f"FILE_ERROR:{idx}:{error_message}\n"
                    
        except Exception as e:
            print(f"整体处理过程错误: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"ERROR:整体处理过程错误: {str(e)}\n"
            
    # 使用stream_with_context包装生成器，确保请求上下文正确维护
    response = Response(stream_with_context(generate_stream()), mimetype='text/plain')
    
    # 注册一个回调，在响应完成后清理临时文件
    @response.call_on_close
    def cleanup():
        print("响应完成，清理临时文件")
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"已删除临时文件: {file_path}")
            except Exception as e:
                print(f"删除临时文件失败: {file_path}, 错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    return response


def ollama_text_stream(input_text, params=None, file_info=None, file_content=None):
    """流式生成摘要文本，逐个token返回"""
    try:
        if not input_text:
            yield "错误：输入文本为空"
            return
            
        if not params:
            params = {}
            
        # 获取参数
        summary_length = params.get('summary_length', 'medium')
        target_language = params.get('target_language', 'chinese')
        summary_style = params.get('summary_style', 'basic')
        focus_area = params.get('focus_area', 'analytical')
        expertise_level = params.get('expertise_level', 'deductive')
        language_style = params.get('language_style', 'precise')
        output_format = params.get('output_format', 'narrative')
        
        # 参数文本化
        summary_length_text = {
            'very_short': '非常简短的摘要（200字左右）',
            'short': '简短摘要（350字左右）',
            'medium': '中等长度摘要（500字左右）',
            'long': '较长摘要（2000字左右）',
            'very_long': '详细摘要（5000字左右）'
        }.get(summary_length, '中等长度摘要（500字左右）')
        
        target_language_text = {
            'chinese': '中文',
            'english': '英文',
            'japanese': '日语',
            'korean': '韩语',
            'french': '法语',
            'german': '德语',
            'russian': '俄语',
            'spanish': '西班牙语'
        }.get(target_language, '中文')
        
        # 不同风格的文本描述
        style_text = {
            'basic': '以基础客观的风格',
            'academic': '以学术论文的风格',
            'business': '以商务报告的风格',
            'technical': '以技术文档的风格',
            'creative': '以创意散文的风格',
            'journalistic': '以新闻报道的风格'
        }.get(summary_style, '以基础客观的风格')
        
        # 关注点文本描述
        focus_text = {
            'comprehensive': '全面涵盖文档各方面内容',
            'analytical': '侧重分析性内容和逻辑关系',
            'comparative': '强调比较性内容和对比关系',
            'critical': '关注评价性内容和批判观点',
            'technical': '突出技术细节和实现方法',
            'practical': '注重实践应用和操作方法'
        }.get(focus_area, '全面涵盖文档各方面内容')
        
        # 专业程度文本描述
        level_text = {
            'introductory': '使用入门级术语解释',
            'intermediate': '使用中级术语和概念',
            'advanced': '使用高级专业术语和深入解释',
            'expert': '使用专家级术语和复杂分析',
            'deductive': '采用演绎推理方式阐述',
            'inductive': '采用归纳推理方式阐述'
        }.get(expertise_level, '使用中级术语和概念')
        
        # 语言风格文本描述
        lang_style_text = {
            'formal': '使用正式语言风格',
            'casual': '使用日常语言风格',
            'simple': '使用简单易懂的语言',
            'precise': '使用精确专业的词汇',
            'persuasive': '使用有说服力的语言',
            'explanatory': '使用解释性的语言'
        }.get(language_style, '使用正式语言风格')
        
        # 输出格式文本描述
        format_text = {
            'paragraph': '输出连续段落式摘要',
            'bullet': '输出要点式摘要',
            'section': '输出分节式摘要',
            'narrative': '输出叙述式摘要',
            'comparative': '输出对比式摘要',
            'analytical': '输出分析式摘要'
        }.get(output_format, '输出连续段落式摘要')
        
        # 选择模型
        model = "huihui_ai/qwen2.5-1m-abliterated"
        
        content_length = len(input_text)
        print(f"输入文本长度: {content_length}")
        
        # 根据输入长度调整上下文窗口
        context_length = min(16384, content_length + 4096)  # 确保上下文大小合理
        
        # 计算需要预测的token数量
        summary_length_map = {
            'very_short': 200,
            'short': 350, 
            'medium': 500,
            'long': 2000,
            'very_long': 5000
        }
        target_word_count = summary_length_map.get(summary_length, 500)
        
        # 中文大约1.5-2个字符对应1个token，再加上一些冗余
        num_predict_tokens = min(int(target_word_count * 3), 16000)  # 限制在模型最大能力范围内
        print(f"目标摘要字数: {target_word_count}, 设置token预测数量: {num_predict_tokens}")
        
        # 构建提示语
        prompt = f"""请你是一个专业的文档摘要分析师。根据以下文档，生成一个{summary_length_text}，使用{target_language_text}，{style_text}。
{focus_text}，{level_text}，{lang_style_text}，{format_text}。

【重要长度要求】：必须严格生成长度为{target_word_count}字的摘要，不能少于此字数的90%。如果内容不足，请通过添加更多细节、解释和具体案例来达到要求字数。

【格式要求】：
- 首先输出标记[KEYWORDS]，然后在下一行列出4-6个关键词，以竖线(|)分隔
- 然后输出标记[SUMMARY]，之后开始你的摘要正文
- 摘要应当包含充分的解释、分析和支持性细节，以达到{target_word_count}字

请使用以下思维链步骤来生成高质量摘要：

步骤1：深入阅读文档，确定文档的主题、目的和主要观点。
步骤2：提取关键信息和中心思想，包括核心主题、主要论点、关键证据和结论。
步骤3：分析文档的结构和逻辑流程，确定各个部分之间的关系。
步骤4：构建一个连贯、完整的摘要框架，确保能支撑{target_word_count}字的详细内容。
步骤5：生成详细摘要，确保：
   - 每个关键点都有充分展开，提供足够的事实和数据支持
   - 对复杂概念进行深入解释和分析
   - 添加具体案例和应用场景说明
   - 提供必要的背景信息和上下文
   - 确保总字数达到{target_word_count}字

==== 文档内容 ====
{input_text}
==== 文档内容结束 ====

首先输出[KEYWORDS]和关键词，然后输出[SUMMARY]和摘要正文。务必确保摘要长度达到{target_word_count}字："""

        # 创建客户端
        client = Client(host='http://localhost:11434')
        print("开始调用大模型生成摘要")

        # 调用模型API - 流式响应
        response_stream = client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={
                'num_predict': num_predict_tokens,  # 使用前面计算的预测token数量
                'temperature': 0.8,  # 适当提高温度以获得更多样化的输出
                'top_p': 0.9,
                'num_ctx': context_length,  # 使用前面计算的上下文长度
                'stop': None
            }
        )
        
        print("已开始流式响应")
        
        # 直接发送一个换行，确保前端开始显示
        yield "\n"
        
        # 用于收集完整响应
        full_response = ""
        
        for response_chunk in response_stream:
            if 'response' in response_chunk:
                token = response_chunk['response']
                full_response += token
                
                # 直接流式输出每个token
                yield token
        
        # 使用process_response函数处理完整响应，提取关键词和摘要
        keywords_list, summary_text = process_response(full_response, target_word_count)
        keywords = "|".join(keywords_list)
        
        # 检查摘要长度是否达到要求，如果未达到，进行补充生成
        current_length = len(summary_text)
        min_required_length = int(target_word_count * 0.9)  # 设置最低要求为目标字数的90%
        
        print(f"当前摘要长度: {current_length}, 最低要求长度: {min_required_length}")
        
        if current_length < min_required_length:
            print(f"摘要长度不足，开始补充生成以达到至少 {min_required_length} 字...")
            
            # 构建补充生成的提示语
            expansion_prompt = f"""你是一位专业的文档摘要专家，请对以下摘要进行扩充，使其达到{target_word_count}字左右。

【扩充要求】:
1. 保持摘要的原有结构和逻辑
2. 对每个要点进行更详细的阐述，增加具体例子和细节
3. 补充必要的背景信息和上下文
4. 确保扩充内容与原摘要保持一致的风格和专业度
5. 避免生硬拼接，使扩充后的摘要流畅自然
6. 摘要总长度必须达到至少{min_required_length}字

【原摘要】:
{summary_text}

请直接输出扩充后的完整摘要，不要包含其他内容:
"""

            print("调用API进行摘要扩充...")
            try:
                # 非流式调用，直接获取完整扩充结果
                expansion_response = client.generate(
                    model=model,
                    prompt=expansion_prompt,
                    stream=False,
                    options={
                        'num_predict': num_predict_tokens,
                        'temperature': 0.8,
                        'top_p': 0.9,
                        'num_ctx': context_length,
                        'stop': None
                    }
                )
                
                if expansion_response and 'response' in expansion_response:
                    expanded_text = expansion_response['response'].strip()
                    expanded_length = len(expanded_text)
                    
                    if expanded_length > current_length:
                        print(f"摘要扩充成功: 从 {current_length} 字增加到 {expanded_length} 字")
                        # 更新摘要文本
                        summary_text = expanded_text
                        # 输出扩充的文本
                        yield "\n\n--- 摘要补充内容 ---\n\n"
                        yield expanded_text[current_length:]
                    else:
                        print(f"扩充未增加长度: 原长度 {current_length}, 新长度 {expanded_length}")
                else:
                    print(f"扩充摘要API调用未返回有效响应")
            except Exception as e:
                print(f"扩充摘要时出错: {str(e)}")
                traceback.print_exc()
        else:
            print(f"摘要长度已达到要求: {current_length} >= {min_required_length}")
            
        # 保存到数据库
        print("开始保存摘要到数据库...")
        
        print("\n=== 开始保存摘要到数据库 ===")
        print(f"摘要长度: {current_length}")
        
        # 如果有parameters参数，创建DocumentSummary记录
        if params and file_info:
            try:
                # 这里是原有的保存文档摘要的逻辑
                # 保留不变...
                pass
            except Exception as e:
                print(f"保存DocumentSummary时出错: {str(e)}")
                traceback.print_exc()
        else:
            # 处理普通摘要保存逻辑，保存到Summary表
            # 这里假设您的应用中有一个Summary表
            try:
                summary_record = Summary(
                    original_text=input_text,
                    summary_text=summary_text,
                    keywords=keywords,
                    model=model,
                    target_language=target_language,
                    target_length=summary_length,
                    focus_areas=",".join(params.get('focus_area', [])) if params else "",
                    level=expertise_level if params else "",
                    language_style=language_style if params else "",
                    style=summary_style if params else "",
                    format=output_format if params else "",
                    timestamp=datetime.now()
                )
                
                db.session.add(summary_record)
                db.session.commit()
                print(f"摘要保存成功，ID: {summary_record.id}")
            except Exception as e:
                print(f"保存Summary时出错: {str(e)}")
                traceback.print_exc()
    except Exception as e:
        error_msg = f"生成摘要时发生错误: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        yield error_msg

@app.route('/download/<int:summary_id>')
def download_document(summary_id):
    """下载原始文档"""
    try:
        # 获取摘要记录
        summary = DocumentSummary.query.get_or_404(summary_id)
        
        # 首先检查是否有原始文本
        if summary.original_text:
            print(f"使用存储的原始文本下载 - summary_id: {summary_id}")
            content = summary.original_text
            content_type = 'text/plain; charset=utf-8'
        else:
            # 如果没有原始文本，尝试获取文件内容
            print(f"尝试获取文件内容下载 - summary_id: {summary_id}")
            content = get_file_content(summary_id)
            if content is None:
                return jsonify({'error': '文件内容不存在'}), 404
            content_type = summary.mime_type or 'application/octet-stream'
        
        # 使用原始文件名或显示文件名
        filename = summary.original_filename or summary.display_filename or summary.file_name
        
        # 创建响应
        response = make_response(content)
        response.headers['Content-Type'] = content_type
        response.headers['Content-Disposition'] = f'attachment; filename={quote(filename)}'
        return response
        
    except Exception as e:
        print(f"下载文件错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/preview/<int:summary_id>')
def preview_document(summary_id):
    """在浏览器中预览文档"""
    try:
        # 获取摘要记录
        summary = DocumentSummary.query.get_or_404(summary_id)
        
        # 使用原始文件名或显示文件名
        filename = summary.original_filename or summary.display_filename or summary.file_name
        
        # 获取文件类型
        file_type = ''
        if '.' in filename:
            file_type = filename.rsplit('.', 1)[1].lower()
        
        # 处理AJAX请求 - 用于分页加载
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            page = request.args.get('page', 1, type=int)
            page_size = request.args.get('page_size', 5000, type=int)  # 默认每页5000个字符
            
            # 获取文本内容 - 首先尝试使用存储的原始文本
            text_content = summary.original_text
            
            # 如果原始文本不存在，则尝试从文件内容中提取
            if not text_content:
                print(f"没有存储的原始文本，尝试从文件内容提取文本 - summary_id: {summary_id}")
                # 获取二进制文件内容
                file_content = get_file_content(summary_id)
                if not file_content:
                    return jsonify({
                        'error': '无法获取文件内容，文件可能已损坏或不存在',
                        'current_page': 0,
                        'total_pages': 0,
                        'has_more': False,
                        'content': ''
                    })
                
                # 根据文件类型提取文本内容
                try:
                    # 为PDF处理创建临时文件
                    if file_type == 'pdf':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(file_content)
                            temp_path = temp_file.name
                        
                        try:
                            # 使用PyMuPDF处理PDF
                            text_content = ""
                            doc = fitz.open(temp_path)
                            for page_num in range(len(doc)):
                                text_content += doc[page_num].get_text()
                            doc.close()
                        finally:
                            # 确保临时文件被删除
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    elif file_type == 'docx':
                        # 处理DOCX文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                            temp_file.write(file_content)
                            temp_path = temp_file.name
                        
                        try:
                            doc = Document(temp_path)
                            text_content = "\n".join([para.text for para in doc.paragraphs])
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    else:
                        # 对于TXT和其他文本文件，尝试直接解码
                        try:
                            text_content = file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                text_content = file_content.decode('latin-1')
                            except Exception:
                                text_content = "无法解码文件内容，请下载原文查看。"
                except Exception as e:
                    print(f"提取文本内容出错: {str(e)}")
                    text_content = f"无法提取文本内容，错误: {str(e)}"
                
                # 如果成功提取了文本，更新数据库中的原始文本
                if text_content and text_content != "无法解码文件内容，请下载原文查看。" and not text_content.startswith("无法提取文本内容"):
                    try:
                        print(f"更新数据库中的原始文本 - summary_id: {summary_id}")
                        summary.original_text = text_content
                        db.session.commit()
                        print(f"成功更新原始文本 - summary_id: {summary_id}")
                    except Exception as e:
                        print(f"更新原始文本失败: {str(e)}")
                        db.session.rollback()
            
            if not text_content:
                text_content = "文件内容为空或无法提取文本内容。"
            
            # 分页
            start = (page - 1) * page_size
            end = start + page_size
            page_content = text_content[start:end] if start < len(text_content) else ""
            total_pages = (len(text_content) + page_size - 1) // page_size if text_content else 0
            
            # 返回JSON格式的页面内容
            return jsonify({
                'content': page_content,
                'current_page': page,
                'total_pages': total_pages,
                'has_more': page < total_pages,
                'content_length': len(text_content)
            })
        
        # 非AJAX请求 - 返回HTML页面
        # 获取原始文本的前5000个字符作为初始内容
        initial_content = ""
        total_pages = 0
        
        # 尝试获取原始文本
        text_content = summary.original_text
        
        # 如果原始文本不存在，则尝试从文件内容中提取前5000个字符
        if not text_content:
            print(f"初始加载 - 没有存储的原始文本，尝试从文件内容提取 - summary_id: {summary_id}")
            # 获取二进制文件内容
            file_content = get_file_content(summary_id)
            if file_content:
                try:
                    # 为PDF处理创建临时文件
                    if file_type == 'pdf':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(file_content)
                            temp_path = temp_file.name
                        
                        try:
                            # 使用PyMuPDF处理前几页PDF
                            initial_content = ""
                            doc = fitz.open(temp_path)
                            total_pages = len(doc)
                            # 只处理前3页用于初始显示
                            for page_num in range(min(3, total_pages)):
                                initial_content += doc[page_num].get_text()
                            doc.close()
                            
                            # 如果成功提取了文本，更新数据库中的原始文本
                            if initial_content:
                                try:
                                    # 重新打开文档以提取完整文本
                                    doc = fitz.open(temp_path)
                                    full_text = ""
                                    for page_num in range(len(doc)):
                                        full_text += doc[page_num].get_text()
                                    doc.close()
                                    
                                    print(f"更新数据库中的原始文本 - summary_id: {summary_id}")
                                    summary.original_text = full_text
                                    db.session.commit()
                                    print(f"成功更新原始文本 - summary_id: {summary_id}")
                                    
                                    # 更新total_pages
                                    total_pages = (len(full_text) + 5000 - 1) // 5000
                                except Exception as e:
                                    print(f"更新原始文本失败: {str(e)}")
                                    db.session.rollback()
                        finally:
                            # 确保临时文件被删除
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    elif file_type == 'docx':
                        # 处理DOCX文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                            temp_file.write(file_content)
                            temp_path = temp_file.name
                        
                        try:
                            doc = Document(temp_path)
                            # 获取所有段落
                            paragraphs = [para.text for para in doc.paragraphs]
                            # 只获取部分段落用于初始显示
                            total_pages = (len(paragraphs) + 20 - 1) // 20  # 假设每页20个段落
                            initial_content = "\n".join(paragraphs[:60])  # 取前60个段落
                            
                            # 如果成功提取了文本，更新数据库中的原始文本
                            if paragraphs:
                                try:
                                    full_text = "\n".join(paragraphs)
                                    print(f"更新数据库中的原始文本 - summary_id: {summary_id}")
                                    summary.original_text = full_text
                                    db.session.commit()
                                    print(f"成功更新原始文本 - summary_id: {summary_id}")
                                    
                                    # 更新total_pages
                                    total_pages = (len(full_text) + 5000 - 1) // 5000
                                except Exception as e:
                                    print(f"更新原始文本失败: {str(e)}")
                                    db.session.rollback()
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    else:
                        # 对于TXT和其他文本文件
                        try:
                            text_content = file_content.decode('utf-8')
                            total_pages = (len(text_content) + 5000 - 1) // 5000
                            initial_content = text_content[:5000]
                            
                            # 更新数据库中的原始文本
                            try:
                                print(f"更新数据库中的原始文本 - summary_id: {summary_id}")
                                summary.original_text = text_content
                                db.session.commit()
                                print(f"成功更新原始文本 - summary_id: {summary_id}")
                            except Exception as e:
                                print(f"更新原始文本失败: {str(e)}")
                                db.session.rollback()
                        except UnicodeDecodeError:
                            try:
                                text_content = file_content.decode('latin-1')
                                total_pages = (len(text_content) + 5000 - 1) // 5000
                                initial_content = text_content[:5000]
                                
                                # 更新数据库中的原始文本
                                try:
                                    print(f"更新数据库中的原始文本 - summary_id: {summary_id}")
                                    summary.original_text = text_content
                                    db.session.commit()
                                    print(f"成功更新原始文本 - summary_id: {summary_id}")
                                except Exception as e:
                                    print(f"更新原始文本失败: {str(e)}")
                                    db.session.rollback()
                            except Exception:
                                initial_content = "无法解码文件内容，请下载原文查看。"
                except Exception as e:
                    print(f"提取初始内容出错: {str(e)}")
                    initial_content = f"无法提取文本内容，错误: {str(e)}"
            else:
                initial_content = "无法获取文件内容，文件可能已损坏或不存在。"
        else:
            # 使用存储的原始文本
            total_pages = (len(text_content) + 5000 - 1) // 5000
            initial_content = text_content[:5000]
        
        return render_template('preview.html', 
                               summary=summary, 
                               file_type=file_type, 
                               total_pages=total_pages,
                               initial_content=initial_content,
                               filename=filename)
        
    except Exception as e:
        print(f"预览文件错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': str(e)}), 500
        # 对于HTML页面请求，返回错误页面
        return render_template('error.html', error=str(e)), 500

class RAGTools:
    """RAG工具类，用于处理文档的向量存储和检索"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="snowflake-arctic-embed2",
            base_url="http://localhost:11434"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.persist_directory = "chroma_db"
        
    def create_vector_store(self, texts, doc_id, metadata=None):
        """创建向量存储"""
        try:
            # 分割文本
            if isinstance(texts, str):
                texts = [texts]
            
            print(f"开始为文档ID {doc_id} 创建向量存储...")
            print(f"输入文本数量: {len(texts)}, 类型: {type(texts[0] if texts else None)}")
            
            if not texts or len(texts) == 0 or not texts[0] or len(texts[0].strip()) == 0:
                print("错误: 输入文本为空，无法创建向量存储")
                return False
            
            # 为每个文本块添加元数据
            documents = []
            print(f"正在分割文本为小块...")
            for i, text in enumerate(texts):
                chunks = self.text_splitter.split_text(text)
                print(f"文本 {i} 分割为 {len(chunks)} 个块")
                for j, chunk in enumerate(chunks):
                    doc_metadata = {
                        'doc_id': doc_id,
                        'chunk_id': i,
                        'chunk_index': j,
                        'source': 'content'
                    }
                    if metadata:
                        doc_metadata.update(metadata)
                    documents.append({'page_content': chunk, 'metadata': doc_metadata})
            
            print(f"总共创建了 {len(documents)} 个文档对象")
            if len(documents) == 0:
                print("警告: 没有创建任何文档对象，可能是输入文本过短或分割问题")
                return False
            
            # 创建或获取向量存储
            collection_name = f"doc_{doc_id}"
            print(f"使用集合名称: {collection_name}, 持久化目录: {self.persist_directory}")
            
            # 检查持久化目录是否存在，如果不存在则创建
            if not os.path.exists(self.persist_directory):
                print(f"创建持久化目录: {self.persist_directory}")
                os.makedirs(self.persist_directory)
                
            try:
                print(f"正在创建/连接Chroma集合...")
                db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                print(f"成功创建/连接到Chroma集合: {collection_name}")
            except Exception as e:
                print(f"创建/连接Chroma集合时出错: {str(e)}")
                raise
            
            # 添加文档
            texts = [doc['page_content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            print(f"开始添加 {len(texts)} 个文本到向量存储")
            try:
                print(f"正在调用Chroma.add_texts添加文本...")
                db.add_texts(texts=texts, metadatas=metadatas)
                print(f"成功添加文本到向量存储")
            except Exception as e:
                print(f"添加文本到向量存储时出错: {str(e)}")
                raise
            
            # 注意：在较新版本的langchain_chroma中，Chroma对象不再需要显式调用persist()方法
            # 数据会自动持久化到指定的persist_directory目录
            print(f"向量存储已自动持久化到磁盘 (目录: {self.persist_directory})")
            
            print(f"文档ID {doc_id} 的向量存储创建完成")
            return True
            
        except Exception as e:
            print(f"创建向量存储失败: {str(e)}")
            traceback.print_exc()
            return False

    def semantic_search(self, query, doc_id, top_k=5):
        """语义搜索"""
        try:
            # 获取向量存储
            collection_name = f"doc_{doc_id}"
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # 创建检索器
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            # 执行检索
            docs = retriever.get_relevant_documents(query)
            
            # 格式化结果
            results = []
            for doc in docs:
                results.append({
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0
                })
            
            return results
            
        except Exception as e:
            print(f"语义搜索失败: {str(e)}")
            traceback.print_exc()
            return []
            
    def generate_rag_response(self, query, doc_id, streaming=False):
        """生成RAG响应"""
        try:
            # 获取向量存储
            collection_name = f"doc_{doc_id}"
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # 创建检索器
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # 创建LLM
            llm = Ollama(
                model="huihui_ai/qwen2.5-1m-abliterated",
                base_url="http://localhost:11434",
                streaming=streaming,
                callbacks=[StreamingStdOutCallbackHandler()] if streaming else None
            )
            
            # 创建提示模板
            template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文: {context}

问题: {question}

请按照以下格式提供答案：
1. 直接回答问题
2. 解释你的答案
3. 引用相关的上下文内容来支持你的答案

答案："""

            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )
            
            # 创建问答链
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            
            # 执行问答
            response = qa_chain({"query": query})
            
            return {
                'answer': response['result'],
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in response['source_documents']
                ]
            }
            
        except Exception as e:
            print(f"生成RAG响应失败: {str(e)}")
            traceback.print_exc()
            return None
            
    def create_conversational_chain(self, doc_id):
        """创建对话链"""
        try:
            # 获取向量存储
            collection_name = f"doc_{doc_id}"
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # 创建检索器
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # 创建LLM
            llm = Ollama(
                model="huihui_ai/qwen2.5-1m-abliterated",
                base_url="http://localhost:11434"
            )
            
            # 创建记忆组件
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # 创建对话链
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )
            
            return chain
            
        except Exception as e:
            print(f"创建对话链失败: {str(e)}")
            traceback.print_exc()
            return None

# 创建RAG工具实例
rag_tools = RAGTools()

@app.route('/chat/<int:doc_id>', methods=['POST'])
def chat_with_document(doc_id):
    """与文档进行对话"""
    try:
        data = request.get_json()
        query = data.get('query')
        chat_history = data.get('chat_history', [])
        
        if not query:
            return jsonify({
                'success': False,
                'error': '请输入问题'
            }), 400
            
        # 创建或获取对话链
        chain = rag_tools.create_conversational_chain(doc_id)
        if not chain:
            return jsonify({
                'success': False,
                'error': '创建对话链失败'
            }), 500
            
        # 执行对话
        response = chain({
            "question": query,
            "chat_history": chat_history
        })
        
        return jsonify({
            'success': True,
            'answer': response['answer'],
            'source_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in response.get('source_documents', [])
            ],
            'chat_history': chat_history + [(query, response['answer'])]
        })
        
    except Exception as e:
        print(f"文档对话错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def clean_orphaned_vector_stores():
    """清理不再有对应数据库记录的向量存储"""
    try:
        print("\n=== 开始清理孤立的向量存储 ===")
        
        # 获取数据库中所有有向量存储的文档ID
        docs_with_vectors = DocumentSummary.query.filter_by(has_vector_store=True).all()
        valid_collection_names = set([f"doc_{doc.id}" for doc in docs_with_vectors])
        print(f"数据库中记录的有效向量存储集合数: {len(valid_collection_names)}")
        
        # 获取磁盘上的所有集合
        try:
            from chromadb import PersistentClient
            persist_directory = "chroma_db"
            
            # 确保目录存在
            if not os.path.exists(persist_directory):
                print(f"向量存储目录不存在: {persist_directory}")
                return {
                    'success': True,
                    'message': '向量存储目录不存在，无需清理',
                    'removed_collections': []
                }
            
            chroma_client = PersistentClient(path=persist_directory)
            all_collections = chroma_client.list_collections()
            
            # 兼容ChromaDB不同版本的API
            try:
                # ChromaDB v0.6.0+版本，list_collections直接返回集合名称列表
                all_collection_names = set(all_collections)
                print(f"ChromaDB v0.6.0+: 获取到 {len(all_collection_names)} 个集合名称")
            except Exception:
                try:
                    # 如果不是直接列表，则尝试获取name属性（旧版本API）
                    all_collection_names = set([col.name for col in all_collections])
                    print(f"旧版ChromaDB: 获取到 {len(all_collection_names)} 个集合名称")
                except Exception as e:
                    print(f"无法获取集合名称: {str(e)}")
                    all_collection_names = set()
                    
            print(f"磁盘上的向量存储集合数: {len(all_collection_names)}")
            
            # 找出孤立的集合（磁盘上有但数据库中没有记录的）
            orphaned_collections = all_collection_names - valid_collection_names
            print(f"发现 {len(orphaned_collections)} 个孤立的向量存储集合")
            
            # 删除孤立的集合
            removed_collections = []
            for collection_name in orphaned_collections:
                try:
                    print(f"删除孤立集合: {collection_name}")
                    chroma_client.delete_collection(collection_name)
                    removed_collections.append(collection_name)
                except Exception as e:
                    print(f"删除集合 {collection_name} 时出错: {str(e)}")
            
            print(f"成功删除 {len(removed_collections)} 个孤立的向量存储集合")
            return {
                'success': True,
                'message': f'成功删除 {len(removed_collections)} 个孤立的向量存储集合',
                'removed_collections': list(removed_collections)
            }
            
        except Exception as e:
            print(f"获取或删除向量存储集合时出错: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    except Exception as e:
        print(f"清理孤立向量存储时出错: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/admin/clean_vector_stores', methods=['POST'])
def handle_clean_vector_stores():
    """API端点：清理孤立的向量存储"""
    try:
        result = clean_orphaned_vector_stores()
        return jsonify(result)
    except Exception as e:
        print(f"清理向量存储API错误: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_response(response: str, target_word_count: int) -> tuple:
    """处理模型的回复，提取关键词和摘要内容"""
    # 初始化默认值
    keywords = []
    summary_text = ""
    
    # 尝试提取关键词
    keywords_match = re.search(r'\[KEYWORDS\](.*?)(?=\[SUMMARY\]|\Z)', response, re.DOTALL)
    if keywords_match:
        keywords_text = keywords_match.group(1).strip()
        # 提取使用竖线分隔的关键词
        if '|' in keywords_text:
            keywords = [k.strip() for k in keywords_text.split('|') if k.strip()]
        # 如果没有用竖线分隔，尝试按行分割
        elif '\n' in keywords_text:
            keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
        # 如果还是空，尝试按空格分割取有意义的词
        else:
            potential_keywords = [k.strip() for k in keywords_text.split() if len(k.strip()) > 1]
            keywords = potential_keywords[:6]  # 最多取6个关键词
    
    # 如果没有找到关键词或关键词为空，使用默认关键词
    if not keywords:
        print("未检测到关键词，使用默认关键词")
        keywords = ["摘要", "文档", "内容", "分析"]
    
    # 尝试提取摘要内容
    summary_match = re.search(r'\[SUMMARY\](.*)', response, re.DOTALL)
    
    if summary_match:
        print("找到[SUMMARY]标记")
        summary_text = summary_match.group(1).strip()
    else:
        print("没有找到明确的[SUMMARY]标记")
        # 如果找到了关键词但没有摘要标记，尝试使用关键词后的所有内容作为摘要
        if keywords_match:
            remaining_text = response[keywords_match.end():].strip()
            if remaining_text:
                print("使用关键词后的内容作为摘要")
                summary_text = remaining_text
        
        # 如果上述方法都失败，使用整个响应作为摘要
        if not summary_text:
            print("使用完整响应作为摘要")
            summary_text = response.strip()
    
    # 检查摘要长度是否足够
    word_count = len(summary_text)
    min_required = int(target_word_count * 0.9)  # 至少达到目标长度的90%
    
    print(f"摘要长度: {word_count}, 目标长度: {target_word_count}, 最小要求: {min_required}")
    
    if word_count < min_required:
        print(f"警告: 摘要长度({word_count})未达到最小要求({min_required})")
    
    return keywords, summary_text

if __name__ == '__main__':
    with app.app_context():
        # 只在表不存在时创建表
        db.create_all()
        # 初始化管理员账户
        init_admin()
        print("数据库初始化完成")
    # 修改端口为5001或8080等非常用端口，并明确指定主机地址
    app.run(debug=True, host='127.0.0.1', port=8080)
