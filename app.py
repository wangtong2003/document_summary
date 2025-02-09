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
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:20030221@localhost/document_summary?charset=utf8mb4'
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
    keywords = db.Column(db.String(255))
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
        print(f"文件信息: {file_info}")
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
                    content_store_path, summary_store_path = create_hybrid_vector_store(
                        new_summary.original_text,
                        new_summary.summary_text,
                        new_summary.id
                    )
                    
                    # 将向量存储路径保存到数据库
                    new_summary.content_vectors = content_store_path.encode()
                    new_summary.summary_vectors = summary_store_path.encode()
                    db.session.commit()
                    print(f"向量存储创建成功，保存在 {content_store_path} 和 {summary_store_path}")
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
        
        # 首先尝试使用 PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"PDF文件共有 {num_pages} 页")
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                        print(f"成功读取第 {page_num + 1} 页，提取到 {len(page_text)} 个字符")
                    except Exception as e:
                        print(f"PyPDF2读取第 {page_num + 1} 页时出错: {str(e)}")
                        continue
            
            if text.strip():
                return text
                
            # 如果PyPDF2提取的文本为空，尝试使用fitz (PyMuPDF)
            print("PyPDF2提取的文本为空，尝试使用PyMuPDF...")
            raise Exception("PyPDF2提取的文本为空")
                
        except Exception as e:
            print(f"PyPDF2处理失败，尝试使用PyMuPDF: {str(e)}")
            # 如果PyPDF2失败，使用fitz (PyMuPDF)作为备选
            with fitz.open(file_path) as doc:
                num_pages = doc.page_count
                print(f"PDF文件共有 {num_pages} 页 (PyMuPDF)")
                
                for page_num in range(num_pages):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            text += page_text + "\n\n"
                        print(f"成功读取第 {page_num + 1} 页，提取到 {len(page_text)} 个字符")
                    except Exception as e:
                        print(f"PyMuPDF读取第 {page_num + 1} 页时出错: {str(e)}")
                        continue
        
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
        text = []
        
        # 提取段落文本
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
                
        # 提取表格文本
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text.append(cell.text)
        
        # 合并所有文本，使用双换行符分隔
        result = '\n\n'.join(text)
        
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
        system_prompt = """You are a powerful AI assistant capable of reading text content and summarizing it. 
        Please follow these basic requirements:

        [BASIC REQUIREMENTS]
        - Summary Length: 约500字
        - Target Language: chinese
        - Keywords: Generate exactly 4 keywords that best represent the main topics of the text

        [OUTPUT FORMAT]
        Please format your response as follows:

        [KEYWORDS]
        keyword1|keyword2|keyword3|keyword4

        [SUMMARY]
        Your summary text here...

        [INSTRUCTIONS]
        1. First output exactly 4 keywords separated by | character
        2. Then output the summary after [SUMMARY] marker
        3. Keywords should be concise and representative
        4. Write everything in chinese
        """
        
        content = f"""
        ARTICLE_TO_SUMMARY:
        <ARTICLE>
        {text}
        </ARTICLE>
        """
        
        print("调用程 Ollama API...")
        response = client.generate(
            model='qwen2.5:3b',
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
    try:
        client = Client(host='http://localhost:11434')
        
        # 获取参数
        summary_length = params.get('summary_length', 'medium')
        target_language = params.get('target_language', 'chinese')
        summary_style = params.get('summary_style', '')
        focus_area = params.get('focus_area', '')
        expertise_level = params.get('expertise_level', '')
        language_style = params.get('language_style', '')
        
        # 更新字数映射
        summary_length_map = {
            'very_short': '100字',
            'medium': '500字',
            'long': '2000字',
            'very_long': '5000字'
        }
        
        target_length = summary_length_map.get(summary_length, '500字')
        target_word_count = int(target_length.replace('字', ''))

        # 使用思维链提示词
        chain_of_thought_prompt = f"""你是一个专业的文档摘要专家。请使用以下步骤分析文档并生成摘要：

        [分析步骤]
        1. 文档结构分析
        - 识别文档的主要部分和层次结构
        - 找出每个部分的主题句和关键论点
        
        2. 核心内容提取
        - 提取文档的主要论点和观点
        - 识别重要的数据、事实和证据
        - 找出作者的立场和结论
        
        3. 关键信息整合
        - 将相关信息按逻辑关系组织
        - 确保不同部分之间的连贯性
        - 保留最具代表性的例子和论据
        
        4. 摘要生成
        - 字数要求：{target_word_count}字(允许±5%)
        - 语言风格：{language_style}
        - 专业程度：{expertise_level}
        - 重点关注：{focus_area}
        - 表达方式：{summary_style}
        
        [输出格式]
        请按以下格式输出你的分析和摘要：

        [文档分析]
        1. 文档结构：
        (分点说明文档的结构特点)

        2. 主要论点：
        (列出核心论点和观点)

        3. 重要证据：
        (列出关键的数据和证据)

        [关键词]
        keyword1|keyword2|keyword3|keyword4

        [摘要]
        (生成最终的摘要文本)

        注意：
        1. 分析过程要清晰可见
        2. 确保摘要完整、准确、连贯
        3. 严格控制最终摘要的字数
        4. 使用恰当的过渡词连接各部分
        5. 保持客观专业的语言风格
        """

        print("调用 Ollama API 生成摘要...")
        response = client.generate(
            model='qwen2.5:3b',
            prompt=chain_of_thought_prompt + f"\n\n文档内容:\n{input_text}",
            stream=False,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': target_word_count * 4  # 确保有足够的生成长度
            }
        )

        if not response or 'response' not in response:
            raise Exception("API 返回的数据格式不正确")
            
        result = response['response']
        print("摘要生成完成")

        # 提取关键词和摘要文本
        try:
            # 提取关键词
            keywords_match = re.search(r'\[关键词\](.*?)\[摘要\]', result, re.DOTALL)
            keywords = keywords_match.group(1).strip() if keywords_match else ''
            
            # 提取摘要文本
            summary_match = re.search(r'\[摘要\](.*?)$', result, re.DOTALL)
            summary_text = summary_match.group(1).strip() if summary_match else result.strip()
            
            # 更新文件信息
            if file_info is not None:
                file_info['keywords'] = keywords
            
            # 保存到数据库
            if file_info:
                save_summary_to_db(file_info, summary_text, params, file_content)
            
            return summary_text
            
        except Exception as e:
            print(f"提取摘要内容时出错: {str(e)}")
            # 如果提取失败，返回原始响应
            if file_info:
                save_summary_to_db(file_info, result, params, file_content)
            return result

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
    """使用 Ollama API 进行文档主题分析"""
    try:
        print("\n=== 开始文档主题分析 ===")
        print(f"原始文本长度: {len(text)} 字符")
        
        client = Client(host='http://localhost:11434')
        
        # 构建提示词
        system_prompt = """你是一个专业的文档主题分析专家。请严格按照以下格式分析文档主题：

        [输出格式示例]
        {
            "topics": [
                {
                    "title": "深度学习技术",
                    "weight": 35,
                    "keywords": ["神经网络", "机器学习", "算法"],
                    "description": "探讨深度学习在图像识别中的应用"
                },
                {
                    "title": "系统实现",
                    "weight": 30,
                    "keywords": ["架构设计", "模块开发", "接口"],
                    "description": "描述系统的具体实现方法和技术架构"
                }
            ]
        }

        [分析要求]
        1. 识别3-5个主要主题
        2. 为每个主题提供简短标题
        3. 计算每个主题的权重占比（总和为100%）
        4. 为每个主题提供3-5个关键词
        5. 为每个主题提供一句话描述

        [严格要求]
        1. 必须使用标准JSON格式输出
        2. 必须包含且仅包含示例中的字段
        3. 权重必须是数字，不要带百分号
        4. 不要添加任何其他内容或说明
        5. 确保JSON格式的完整性和正确性

        请直接输出JSON，不要包含任何其他内容。如果你理解了这些要求，请直接输出符合要求的JSON，不要有任何前缀或后缀说明。
        """
        
        content = f"""
        需要分析的文本内容：
        <文本>
        {text}
        </文本>
        """
        
        print("调用 Ollama API...")
        response = client.generate(
            model='qwen2.5:3b',
            prompt=system_prompt + content,
            stream=False
        )
        
        if not response or 'response' not in response:
            raise Exception("API 返回的数据格式不正确")
            
        # 提取 JSON 字符串
        response_text = response['response']
        print("原始响应:", response_text[:200])
        
        try:
            # 首先尝试直接解析
            analysis_result = json.loads(response_text)
        except json.JSONDecodeError:
            print("直接解析失败，尝试提取和修复JSON...")
            try:
                # 查找第一个 { 和最后一个 }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    # 如果找不到完整的JSON，尝试从文本中提取信息构建JSON
                    print("尝试从文本中提取信息构建JSON...")
                    
                    # 使用正则表达式提取可能的主题信息
                    topics = []
                    
                    # 尝试匹配标题和描述
                    title_matches = re.finditer(r'[一二三四五六1234567]、([^，。\n]+)', response_text)
                    descriptions = re.finditer(r'[:：]([^。\n]+)[。\n]', response_text)
                    
                    # 提取关键词
                    keyword_matches = re.finditer(r'[【\[](.*?)[】\]]', response_text)
                    keywords = [m.group(1).split('、') for m in keyword_matches]
                    
                    # 构建主题列表
                    titles = [m.group(1) for m in title_matches]
                    descs = [m.group(1) for m in descriptions]
                    
                    # 确保至少有3个主题
                    while len(titles) < 3:
                        titles.append(f"主题{len(titles)+1}")
                    
                    # 构建主题列表
                    total_topics = min(5, len(titles))  # 最多5个主题
                    base_weight = 100 // total_topics
                    remaining_weight = 100 - (base_weight * total_topics)
                    
                    for i in range(total_topics):
                        topic = {
                            "title": titles[i] if i < len(titles) else f"主题{i+1}",
                            "weight": base_weight + (1 if i < remaining_weight else 0),
                            "keywords": keywords[i][:3] if i < len(keywords) else [f"关键词{j+1}" for j in range(3)],
                            "description": descs[i] if i < len(descs) else f"这是{titles[i]}的描述"
                        }
                        topics.append(topic)
                    
                    analysis_result = {"topics": topics}
                else:
                    # 提取JSON部分
                    json_str = response_text[start_idx:end_idx]
                    # 清理可能的格式问题
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    analysis_result = json.loads(json_str)
            except Exception as e:
                print(f"JSON提取和修复失败: {str(e)}")
                # 创建一个基本的分析结果
                analysis_result = {
                    "topics": [
                        {
                            "title": "主要内容",
                            "weight": 60,
                            "keywords": ["文档", "内容", "分析"],
                            "description": "文档的主要内容和核心观点"
                        },
                        {
                            "title": "补充信息",
                            "weight": 40,
                            "keywords": ["细节", "说明", "补充"],
                            "description": "文档的补充信息和详细说明"
                        }
                    ]
                }
        
        # 验证和规范化结果
        if 'topics' not in analysis_result:
            raise Exception("返回的数据缺少 topics 字段")
            
        # 确保权重总和为 100%
        total_weight = sum(topic['weight'] for topic in analysis_result['topics'])
        if total_weight > 0:
            for topic in analysis_result['topics']:
                topic['weight'] = round((topic['weight'] / total_weight) * 100, 1)
                
        return {
            'success': True,
            'topics': analysis_result['topics']
        }
            
    except Exception as e:
        print(f"主题分析错误: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/analyze_topics/<int:summary_id>')
def analyze_topics(summary_id):
    """获取文档的主题分析"""
    try:
        print(f"\n=== 获取文档主题分析 ID: {summary_id} ===")
        summary = DocumentSummary.query.get_or_404(summary_id)
        
        # 从数据库获取主题分析结果
        if summary.topic_analysis:
            return jsonify({
                'success': True,
                'topics': summary.topic_analysis.get('topics', [])
            })
        
        # 如果数据库中没有主题分析结果，进行分析并保存
        topic_analysis = analyze_document_topics(summary.original_text)
        
        # 更新数据库中的主题分析结果
        summary.topic_analysis = topic_analysis
        db.session.commit()
        
        if not topic_analysis['success']:
            return jsonify({
                'error': topic_analysis.get('error', '主题分析失败')
            }), 500
            
        return jsonify(topic_analysis)
        
    except Exception as e:
        print(f"获取主题分析失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
                            print(f"页面内容预览: {page_text[:200]}...")  # 打印前200个字符
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
                print(f"内容预览: {current_content[:200]}...")  # 打印前200个字符
                
                # 如果是AJAX请求，返回JSON格式的分页内容
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    response_data = {
                        'content': current_content,
                        'has_more': end_page < total_pages,
                        'total_pages': total_pages,
                        'current_page': page
                    }
                    print(f"返回AJAX响应: {str(response_data)[:200]}...")
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
        llm = Ollama(model="qwen2.5:3b")
        
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

def create_hybrid_vector_store(text, summary_text, doc_id):
    """创建混合向量存储（正文和摘要）"""
    try:
        print(f"\n=== 创建混合向量存储 文档ID: {doc_id} ===")
        
        # 获取嵌入模型
        embeddings = get_embeddings_model()
        
        # 1. 处理正文内容
        content_text = text[:5000]  # 只取前5000字进行向量化
        content_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        
        # 分割正文
        content_chunks = content_splitter.split_text(content_text)
        print(f"正文分割为 {len(content_chunks)} 个块")
        
        # 生成正文向量
        content_vectors = []
        for i, chunk in enumerate(content_chunks):
            vector = embeddings.embed_query(chunk)
            content_vectors.append({
                'vector': vector,
                'text': chunk,
                'type': 'content',
                'chunk_idx': i
            })
        
        # 2. 处理摘要内容
        summary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        
        # 分割摘要
        summary_chunks = summary_splitter.split_text(summary_text)
        print(f"摘要分割为 {len(summary_chunks)} 个块")
        
        # 生成摘要向量
        summary_vectors = []
        for i, chunk in enumerate(summary_chunks):
            vector = embeddings.embed_query(chunk)
            summary_vectors.append({
                'vector': vector,
                'text': chunk,
                'type': 'summary',
                'chunk_idx': i
            })
        
        # 序列化向量数据
        content_data = pickle.dumps(content_vectors)
        summary_data = pickle.dumps(summary_vectors)
        
        # 更新数据库
        summary = db.session.get(DocumentSummary, doc_id)
        if summary:
            summary.content_vectors = content_data
            summary.summary_vectors = summary_data
            summary.embedding_model = "snowflake-arctic-embed2"
            summary.chunks_info = {
                "content_chunks": len(content_chunks),
                "summary_chunks": len(summary_chunks),
                "content_chunk_size": 500,
                "summary_chunk_size": 200,
                "content_chunk_overlap": 50,
                "summary_chunk_overlap": 20
            }
            db.session.commit()
            print("向量数据已保存到数据库")
        
        return content_vectors, summary_vectors
        
    except Exception as e:
        print(f"创建混合向量存储失败: {str(e)}")
        traceback.print_exc()
        raise

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
            content_vectors, summary_vectors = create_hybrid_vector_store(
                summary.original_text, 
                summary.summary_text, 
                doc_id
            )
        else:
            # 从数据库加载向量数据
            content_vectors = pickle.loads(summary.content_vectors)
            summary_vectors = pickle.loads(summary.summary_vectors)
        
        # 获取嵌入模型
        embeddings = get_embeddings_model()
        
        # 生成查询向量
        query_vector = embeddings.embed_query(query)
        
        # 计算相似度并排序
        results = []
        
        # 处理正文向量
        for item in content_vectors:
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
        
        # 处理摘要向量
        for item in summary_vectors:
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
        
        # 按得分排序（得分越低越相关）
        results.sort(key=lambda x: x['score'])
        
        return results[:5]  # 返回最相关的5个结果
        
    except Exception as e:
        print(f"混合语义搜索失败: {str(e)}")
        traceback.print_exc()
        raise

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
        llm = Ollama(model="qwen2.5:3b")
        
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
