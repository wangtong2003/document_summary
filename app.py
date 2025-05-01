import pymysql
pymysql.install_as_MySQLdb()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message="As of langchain-core 0.3.0")
warnings.filterwarnings("ignore", message="deprecated", module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
try:
    from langchain.warnings import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*pydantic.*")
from flask import Flask, request, jsonify, Response, render_template, stream_with_context, session, send_from_directory, redirect
from flask_session import Session
from ollama import Client
import os
import fitz
from docx import Document
import markdown
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import uuid
from datetime import datetime, timedelta
from functools import wraps
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re
from flask_migrate import Migrate
import traceback
import time
import sqlalchemy.exc
from sqlalchemy import inspect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
import pickle
from threading import Lock, Thread
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
import unicodedata
import sys
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import math
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from queue import Queue
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('doc_summary')

# 创建一个任务队列用于异步向量化处理
vectorization_queue = Queue()

app = Flask(__name__, static_folder='static')
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
app.config['SECRET_KEY'] = 'a5c9f438d3854c1e9c39b48ae4a5bc7fb62af6239d8745fe8c2baeb4cd4eb812'  # 设置 session 密钥
app.config['SESSION_TYPE'] = 'filesystem'  # 使用文件系统存储 session
app.config['SESSION_FILE_DIR'] = 'flask_session'  # session 文件存储目录
app.config['SESSION_PERMANENT'] = True  # session持久化
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # session 有效期
app.config['SESSION_USE_SIGNER'] = True  # 签名cookie sid，提高安全性
app.config['SESSION_KEY_PREFIX'] = 'docsummary_'  # session前缀

# 初始化 Flask-Session
Session(app)

db = SQLAlchemy(app)
migrate = Migrate(app, db)  # 初始化 Flask-Migrate

# 创建必要的目录
for folder in [UPLOAD_FOLDER, DOCUMENTS_FOLDER, 'flask_session']:
    if not os.path.exists(folder):
        os.makedirs(folder)



# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 调试信息
        print(f"检查登录状态: session={session}")
        if 'user_id' not in session:
            print("用户未登录，重定向到登录页面")
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

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
    content_vectors = db.Column(db.LargeBinary(length=16777215), nullable=True)  # 存储内容的向量数据
    summary_vectors = db.Column(db.LargeBinary(length=16777215), nullable=True)  # 存储摘要的向量数据
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
    
    # 添加用户关联
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('summaries', lazy=True))
    
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
    
    # 新增更实用的输出结构和格式相关字段
    output_structure = db.Column(db.String(20), nullable=True)  # 摘要结构：bullet, outline, paragraph, table, qa
    depth_level = db.Column(db.String(20), nullable=True)  # 层次深度：single, multi, section
    tone = db.Column(db.String(20), nullable=True)  # 语气：formal, conversational, technical, simple
    content_focus = db.Column(db.String(20), nullable=True)  # 内容关注点：topic, chronological, problem_solution, comparative
    citation_style = db.Column(db.String(20), nullable=True)  # 引用格式：none, apa, mla, chicago, harvard
    export_format = db.Column(db.String(20), nullable=True)  # 导出格式：markdown, text, html, docx
    visual_elements = db.Column(db.String(20), nullable=True)  # 是否包含可视元素：none, tables, bullets, numbering
    
    # 保留原有字段
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
        # 获取当前用户ID
        user_id = session.get('user_id')
        if not user_id:
            raise ValueError("用户未登录，无法保存摘要")
            
        print("\n=== 开始保存摘要到数据库 ===")
        print(f"用户ID: {user_id}")
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
        
        # 获取目标语言并生成相应语言的关键词
        target_language = params.get("target_language", "chinese")
        if 'keywords' not in file_info or not file_info['keywords']:
            print(f"为摘要生成 {target_language} 语言的关键词")
            keywords = generate_keywords_with_model(summary_text, target_language)
            file_info['keywords'] = "|".join(keywords)
            print(f"生成的关键词: {file_info['keywords']}")
        
        print(f"原始文件名: {original_filename}")
        print(f"显示文件名: {display_filename}")
        print(f"原始文本长度: {len(original_text) if original_text else 0}")

        try:
            # 检查是否已存在相同文件的摘要 - 增加用户ID过滤
            existing_summary = DocumentSummary.query.filter_by(
                file_hash=file_hash,
                user_id=user_id
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
                
                # 异步创建混合向量存储，改为后台执行
                try:
                    print(f"将文档ID {existing_summary.id} 加入向量化队列")
                    async_create_vector_store(
                        file_info["original_text"], 
                        summary_text, 
                        existing_summary.id
                    )
                except Exception as e:
                    print(f"加入向量化队列失败: {str(e)}")
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
                    user_id=user_id,  # 关联用户ID
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
                
                # 异步创建混合向量存储，改为后台执行
                try:
                    print(f"将文档ID {new_summary.id} 加入向量化队列")
                    print(f"原始文本类型: {type(new_summary.original_text)}, 原始文本长度: {len(new_summary.original_text) if new_summary.original_text else 0}")
                    print(f"摘要文本类型: {type(new_summary.summary_text)}, 摘要文本长度: {len(new_summary.summary_text) if new_summary.summary_text else 0}")
                    async_create_vector_store(
                        new_summary.original_text,
                        new_summary.summary_text,
                        new_summary.id
                    )
                    print("异步向量化任务已创建")
                except Exception as e:
                    print(f"加入向量化队列失败: {str(e)}")
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
@login_required
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
                    
                    # 获取目标语言参数
                    target_language = params.get('target_language', 'chinese')
                    # 进行主题分析，传递目标语言参数
                    topic_analysis = analyze_document_topics(text, target_language)
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
        
        # 获取目标语言
        target_language = params.get('target_language', 'chinese')
        
        # 使用generate_keywords_with_model生成对应语言的关键词
        print(f"正在生成{target_language}语言的关键词...")
        try:
            keyword_list = generate_keywords_with_model(input_text[:3000], target_language)
            keywords = '|'.join(keyword_list)
            print(f"生成的{target_language}关键词: {keywords}")
        except Exception as e:
            print(f"生成关键词失败: {str(e)}")
            print("使用默认关键词")
            keyword_list = get_default_keywords(target_language)
            keywords = '|'.join(keyword_list)
            
        # 确保每个关键词不超过8个字符
        keyword_list = [k[:8] for k in keyword_list]
        keywords = '|'.join(keyword_list)
        print(f"最终关键词: {keywords}")
        
        # 获取参数
        summary_length = params.get('summary_length', 'medium')
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
        summary_prompt = f"""你是一个专业的文档摘要专家。请严格按照思维链方法分析文档并生成高质量{target_language}摘要。

[思维链步骤]
1. 分析：仔细阅读文档，确定主题、目的和主要论点
2. 提取：识别核心概念、关键信息点和重要论述
3. 判断类型：确定文档是学术论文、研究报告、技术文档还是一般文章
4. 分类整理：按照合适的结构组织内容
5. 提炼：从每个部分提取最具代表性的内容
6. 关键词确认：验证已提取的关键词是否准确反映文档核心内容
7. 语言转换：将内容完全转换为{target_language}，保持专业术语的准确性
8. 整合：按照用户指定的参数要求生成最终摘要

[语言控制要求]
1. 摘要必须完全使用{target_language}，不得混合其他语言
2. 遵循{target_language}的语法规则和表达习惯
3. 专业术语需要使用{target_language}的对应表达方式
4. 确保摘要流畅自然，符合{target_language}的阅读习惯
5. 多次检查确保没有混入其他语言的词汇或表达

[输出格式要求]
1. 首先输出 [KEYWORDS] 标记
2. 在其下方输出4个关键词，用竖线(|)分隔
3. 然后输出 [SUMMARY] 标记
4. 最后按照用户指定的格式输出摘要正文（全{target_language}）

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
6. 确保整个摘要100%使用{target_language}，不混入其他语言

[原文内容]
{input_text}

请按照以上步骤和要求生成摘要，确保摘要的长度、风格、格式和内容符合用户指定的所有参数，并且严格使用{target_language}。
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
                        3. 如果超过目标字数20%，需要适当精简
                        4. 必须完全使用{target_language}，不得混入其他语言"""
                    )
                    
                    response = client.generate(
                        model='huihui_ai/qwen2.5-1m-abliterated:latest',
                        prompt=current_prompt,
                        stream=False,
                        options={
                            'num_predict': min(current_num_predict, 16000),  # 不超过模型最大限制
                            'temperature': 0.5 + (0.1 * attempt),  # 降低初始温度以提高一致性，逐步提高创造性
                            'top_p': 0.85,  # 降低随机性，提高输出稳定性
                            'num_ctx': 16384,  # 确保足够上下文窗口
                            'stop': None,
                            'presence_penalty': 0.2  # 增加这个参数可以减少重复，增强语言一致性
                        }
                    )
                    
                    if not response or 'response' not in response:
                        raise Exception("摘要API响应为空或格式错误")
                        
                    summary_text = response['response'].strip()
                    current_length = len(summary_text)
                    
                    # 验证语言纯度
                    if verify_language_purity(summary_text, target_language) is False and attempt < max_retries - 1:
                        print(f"警告：生成的摘要语言不纯，可能混合了其他语言，尝试重新生成...")
                        continue
                    
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
                    
        # 添加语言纯度验证函数
        def verify_language_purity(text, target_language):
            """验证生成文本的语言纯度"""
            # 语言特征字符映射
            language_chars = {
                'chinese': r'[\u4e00-\u9fff]',  # 中文字符
                'english': r'[a-zA-Z]',  # 英文字母
                'japanese': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',  # 日文假名和汉字
                'korean': r'[\uac00-\ud7a3\u1100-\u11ff]',  # 韩文字符
                'russian': r'[\u0400-\u04FF]',  # 西里尔字母
                'german': r'[a-zA-ZäöüÄÖÜß]',  # 德文字母
                'french': r'[a-zA-ZàâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸ]',  # 法文字母
                'spanish': r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]'  # 西班牙文字母
            }
            
            # 容忍的其他语言字符比例
            tolerance = 0.05  # 5%的容忍度
            
            try:
                import re
                
                # 如果目标语言不在映射中，默认通过验证
                if target_language not in language_chars:
                    return True
                
                # 统计目标语言字符数量
                target_chars = len(re.findall(language_chars[target_language], text))
                
                # 统计所有字符（不包括空格和标点符号）
                all_chars = len(re.sub(r'[\s\p{P}]', '', text, flags=re.UNICODE))
                
                if all_chars == 0:
                    return True  # 如果没有有效字符，默认验证通过
                
                # 计算目标语言字符占比
                target_ratio = target_chars / all_chars
                
                # 判断是否达到纯度要求
                return target_ratio >= (1 - tolerance)
            except Exception as e:
                print(f"语言纯度验证出错: {str(e)}")
                return True  # 出错时默认验证通过
                    
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
            session.clear()  # 清除旧session
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session.permanent = True  # 设置为永久session
            
            print(f"用户登录成功: {username}, session_id={request.cookies.get('session')}")
            print(f"Session内容: {session}")
            
            return jsonify({
                'message': '登录成功',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            })
        else:
            return jsonify({'error': '用户名或密码错误'}), 401
            
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
@login_required
def index():
    """主页"""
    # 已登录则显示仪表盘
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    # 如果用户已登录，重定向到主页
    if 'user_id' in session:
        return redirect('/')
    return render_template('login.html')

@app.route('/register')
def register_page_view():
    return render_template('register.html')

@app.route('/summary_library')
@login_required
def summary_library():
    """渲染摘要库页面"""
    
    # 记录请求信息
    referer = request.headers.get('Referer', '')
    print(f"\n=== 访问摘要库 ===")
    print(f"来源: {referer}")
    print(f"用户IP: {request.remote_addr}")
    print(f"用户代理: {request.user_agent}")
    
    # 如果是从预览页面返回的，添加额外处理
    if referer and '/preview/' in referer:
        print(f"检测到从预览页面返回")
        # 这里可以添加特定于预览返回的处理逻辑
    
    # 添加一个变量表示当前使用的embedding模型
    embedding_model_name = "EntropyYue/jina-embeddings-v2-base-zh"
    return render_template('summaries.html', embedding_model=embedding_model_name)

@app.route('/summaries')
@login_required
def get_summaries():
    """获取所有摘要列表（分页）"""
    try:
        print("\n=== 开始获取摘要列表 ===")
        
        # 获取当前登录用户ID
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '用户未登录'}), 401
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # 限制每页数量
        if per_page > 50:
            per_page = 50
            
        # 查询总数 - 只查询当前用户的
        total = DocumentSummary.query.filter_by(user_id=user_id).count()
        
        # 分页查询 - 只查询当前用户的
        pagination = DocumentSummary.query.filter_by(user_id=user_id).order_by(
            DocumentSummary.created_at.desc()
        ).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        summaries = pagination.items
        print(f"用户 {user_id} 查询到 {len(summaries)} 条摘要记录")
        
        results = []
        for summary in summaries:
            # 使用原始文件名作为显示名称
            display_name = summary.original_filename or summary.display_filename or summary.file_name
            
            # 移除文件路径，只显示文件名
            if display_name and ('/' in display_name or '\\' in display_name):
                display_name = os.path.basename(display_name)
            
            # 获取文件扩展名
            file_type = os.path.splitext(display_name)[1].lower().lstrip('.') if display_name else 'unknown'
            
            # 检查文件是否存在
            has_file = bool(summary.file_content) or (hasattr(summary, 'is_chunked') and summary.is_chunked)
            
            # 处理关键词 - 确保始终返回数组格式
            keywords_array = []
            if summary.keywords:
                try:
                    if isinstance(summary.keywords, str) and summary.keywords.strip():
                        # 确保分隔符处理正确
                        if '|' in summary.keywords:
                            keywords_array = [k.strip() for k in summary.keywords.split('|') if k.strip()]
                        else:
                            # 如果没有分隔符，尝试作为单个关键词处理
                            keywords_array = [summary.keywords.strip()]
                        print(f"摘要 ID {summary.id}: 从字符串转换关键词: {summary.keywords} -> {keywords_array}")
                    elif isinstance(summary.keywords, list):
                        keywords_array = [k for k in summary.keywords if k]
                        print(f"摘要 ID {summary.id}: 关键词已经是数组格式: {keywords_array}")
                    else:
                        print(f"摘要 ID {summary.id}: 未知关键词格式: {type(summary.keywords)}, 值: {summary.keywords}")
                        # 尝试强制转换为字符串再处理
                        try:
                            if summary.keywords:
                                str_val = str(summary.keywords)
                                keywords_array = [str_val.strip()]
                                print(f"摘要 ID {summary.id}: 强制转换关键词: {str_val}")
                        except:
                            print(f"摘要 ID {summary.id}: 强制转换失败")
                    
                    # 确保最终结果必须是数组
                    if not isinstance(keywords_array, list):
                        print(f"摘要 ID {summary.id}: 最终转换结果非数组，强制转为空数组")
                        keywords_array = []
                    
                    # 打印最终结果进行确认
                    print(f"摘要 ID {summary.id}: 最终关键词数组: {keywords_array}, 类型: {type(keywords_array)}")
                    
                except Exception as ke:
                    print(f"摘要 ID {summary.id}: 处理关键词时出错: {str(ke)}")
                    keywords_array = []  # 确保出错时返回空数组
            else:
                print(f"摘要 ID {summary.id}: 无关键词数据")
            
            # 如果没有关键词数据，使用大模型生成
            if not keywords_array and summary.summary_text:
                print(f"摘要 ID {summary.id}: 尝试使用大模型生成关键词")
                keywords_array = generate_keywords_with_model(summary.summary_text, summary.target_language)
                print(f"摘要 ID {summary.id}: 大模型生成的关键词: {keywords_array}")
                
                # 如果成功生成关键词，更新数据库
                if keywords_array:
                    try:
                        # 将关键词数组转换为字符串并保存到数据库
                        keywords_str = '|'.join(keywords_array)
                        summary.keywords = keywords_str
                        db.session.commit()
                        print(f"摘要 ID {summary.id}: 已将生成的关键词 {keywords_str} 保存到数据库")
                    except Exception as save_err:
                        print(f"摘要 ID {summary.id}: 保存关键词到数据库失败: {str(save_err)}")
                        db.session.rollback()
            
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
                'keywords': keywords_array  # 使用处理后的数组
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
@login_required
def get_summary_detail(summary_id):
    """获取单摘要详情"""
    try:
        print(f"\n=== 获取摘要详情 ID: {summary_id} ===")
        
        # 获取当前登录用户ID
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '用户未登录'}), 401
            
        # 查找摘要，并确保它属于当前用户
        summary = DocumentSummary.query.filter_by(id=summary_id, user_id=user_id).first()
        
        if not summary:
            print(f"未找到ID为 {summary_id} 的摘要或用户无权访问")
            return jsonify({'error': f'未找到ID为 {summary_id} 的摘要或您无权访问'}), 404
            
        # 使用原始文件名或显示文件名
        display_name = summary.original_filename or summary.display_filename or summary.file_name
        
        # 移除文件路径，只显示文件名
        if display_name and ('/' in display_name or '\\' in display_name):
            display_name = os.path.basename(display_name)
        
        # 处理关键词 - 确保始终返回数组格式
        keywords_array = []
        if summary.keywords:
            try:
                if isinstance(summary.keywords, str) and summary.keywords.strip():
                    # 确保分隔符处理正确
                    if '|' in summary.keywords:
                        keywords_array = [k.strip() for k in summary.keywords.split('|') if k.strip()]
                    else:
                        # 如果没有分隔符，尝试作为单个关键词处理
                        keywords_array = [summary.keywords.strip()]
                    print(f"摘要详情 ID {summary.id}: 从字符串转换关键词: {summary.keywords} -> {keywords_array}")
                elif isinstance(summary.keywords, list):
                    keywords_array = [k for k in summary.keywords if k]
                    print(f"摘要详情 ID {summary.id}: 关键词已经是数组格式: {keywords_array}")
                else:
                    print(f"摘要详情 ID {summary.id}: 未知关键词格式: {type(summary.keywords)}, 值: {summary.keywords}")
                    # 尝试强制转换为字符串再处理
                    try:
                        if summary.keywords:
                            str_val = str(summary.keywords)
                            keywords_array = [str_val.strip()]
                            print(f"摘要详情 ID {summary.id}: 强制转换关键词: {str_val}")
                    except:
                        print(f"摘要详情 ID {summary.id}: 强制转换失败")
                
                # 确保最终结果必须是数组
                if not isinstance(keywords_array, list):
                    print(f"摘要详情 ID {summary.id}: 最终转换结果非数组，强制转为空数组")
                    keywords_array = []
                
                # 打印最终结果进行确认
                print(f"摘要详情 ID {summary.id}: 最终关键词数组: {keywords_array}, 类型: {type(keywords_array)}")
                
            except Exception as ke:
                print(f"摘要详情 ID {summary.id}: 处理关键词时出错: {str(ke)}")
                keywords_array = []  # 确保出错时返回空数组
        else:
            print(f"摘要详情 ID {summary.id}: 无关键词数据")
        
        # 如果没有关键词数据，使用大模型生成
        if not keywords_array and summary.summary_text:
            print(f"摘要详情 ID {summary.id}: 尝试使用大模型生成关键词")
            keywords_array = generate_keywords_with_model(summary.summary_text, summary.target_language)
            print(f"摘要详情 ID {summary.id}: 大模型生成的关键词: {keywords_array}")
            
            
            # 如果成功生成关键词，更新数据库
            if keywords_array:
                try:
                    # 将关键词数组转换为字符串并保存到数据库
                    keywords_str = '|'.join(keywords_array)
                    summary.keywords = keywords_str
                    db.session.commit()
                    print(f"摘要详情 ID {summary.id}: 已将生成的关键词 {keywords_str} 保存到数据库")
                except Exception as save_err:
                    print(f"摘要详情 ID {summary.id}: 保存关键词到数据库失败: {str(save_err)}")
                    db.session.rollback()
        
        result = {
            'id': summary.id,
            'file_name': display_name,  # 使用正确的文件名
            'summary_text': summary.summary_text,
            'original_text': summary.original_text,
            'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': summary.summary_length,
            'target_language': summary.target_language,
            'keywords': keywords_array  # 使用处理后的数组
        }
        return jsonify(result)
        
    except Exception as e:
        print(f"获取摘要详情错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summaries/<int:summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    try:
        # 获取要删除的摘要
        summary = DocumentSummary.query.get(summary_id)
        if not summary:
            error_msg = f"未找到ID为 {summary_id} 的摘要记录"
            print(error_msg)
            return jsonify({'error': error_msg}), 404
            
        # 检查用户权限 - 只有管理员和记录所有者可以删除
        current_user_id = session.get('user_id')
        if not current_user_id:
            return jsonify({'error': '用户未登录'}), 401
            
        current_user = User.query.get(current_user_id)
        if current_user.id != summary.user_id and current_user.role != 'admin':
            return jsonify({'error': '您无权删除此摘要'}), 403
        
        # 如果有向量存储，先删除
        if summary.has_vector_store and summary.chroma_collection:
            try:
                print(f"尝试删除向量存储: {summary.chroma_collection}")
                collection_name = summary.chroma_collection
                persist_directory = "chroma_db"  # 硬编码目录路径，与RAGTools中一致
                
                # 方法1: 首先尝试使用PersistentClient直接删除集合
                try:
                    print("方法1: 使用PersistentClient直接删除集合")
                    client = PersistentClient(path=persist_directory)
                    client.delete_collection(collection_name)
                    print(f"成功通过PersistentClient删除集合 {collection_name}")
                except Exception as e:
                    print(f"通过PersistentClient删除集合失败: {str(e)}")
                    
                    # 方法2: 尝试使用Chroma接口删除
                    try:
                        print("方法2: 使用Chroma接口删除集合")
                        from langchain_chroma import Chroma
                        from langchain_ollama import OllamaEmbeddings
                        
                        embeddings = OllamaEmbeddings(
                            model="huihui_ai/bge-small-zh-v1.5",
                            base_url="http://localhost:11434"
                        )
                        
                        chroma_db = Chroma(
                            persist_directory=persist_directory,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        
                        # 尝试方法2.1: 使用Chroma对象的delete_collection方法
                        if hasattr(chroma_db, 'delete_collection'):
                            chroma_db.delete_collection()
                            print("成功通过Chroma对象删除集合")
                        else:
                            print("Chroma实例没有delete_collection方法")
                            
                            # 尝试方法2.2: 使用Chroma对象的_client的delete_collection方法
                            if hasattr(chroma_db._client, 'delete_collection'):
                                chroma_db._client.delete_collection(collection_name)
                                print("成功通过chroma_db._client删除集合")
                            else:
                                print("chroma_db._client没有delete_collection方法")
                    except Exception as e2:
                        print(f"通过Chroma接口删除集合失败: {str(e2)}")
                
                print(f"已尝试所有可能的方法删除向量存储: {collection_name}")
            except Exception as e:
                print(f"删除向量存储时发生未处理的异常: {str(e)}")
                traceback.print_exc()
        
        # 删除关联的文件映射记录
        try:
            file_mappings = FileMapping.query.filter_by(summary_id=summary_id).all()
            for file_mapping in file_mappings:
                db.session.delete(file_mapping)
            print(f"删除了 {len(file_mappings)} 条文件映射记录")
        except Exception as e:
            print(f"删除文件映射记录时出错: {str(e)}")
        
        # 删除关联的文件块记录
        try:
            chunks = FileChunk.query.filter_by(document_id=summary_id).all()
            for chunk in chunks:
                db.session.delete(chunk)
            print(f"删除了 {len(chunks)} 条文件块记录")
        except Exception as e:
            print(f"删除文件块记录时出错: {str(e)}")
        
        # 最后删除摘要记录
        db.session.delete(summary)
        db.session.commit()
        print(f"成功删除摘要记录: {summary_id}")
        
        return jsonify({'success': True, 'message': f'成功删除摘要(ID: {summary_id})'}), 200
        
    except Exception as e:
        print(f"删除摘要时发生未处理的异常: {str(e)}")
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

def analyze_document_topics(text, target_language=None):
    """使用大模型分析文档主题，返回主题及其关键词
    
    Args:
        text: 要分析的文本内容
        target_language: 目标语言，如果为None则使用中文
    """
    try:
        client = Client(host='http://localhost:11434')
        
        # 限制输入文本长度，避免超出上下文窗口
        text_for_analysis = text[:8000] if len(text) > 8000 else text
        
        # 处理语言选项
        lang_prompts = {
            'chinese': {
                'instruction': '请仔细分析以下文档，提取4-6个主要主题。',
                'json_format': '请以JSON格式返回分析结果，格式如下：',
                'title': '主题名称',
                'weight': '权重',
                'description': '对该主题的简要描述',
                'notes': [
                    '每个主题的权重(weight)之和应为100',
                    '对每个主题提供简短的描述'
                ],
                'output_instruction': '仅返回JSON格式的结果，不要包含任何解释或其他文本。'
            },
            'english': {
                'instruction': 'Carefully analyze the following document, extract 4-6 main topics.',
                'json_format': 'Return the analysis results in JSON format as follows:',
                'title': 'Topic Name',
                'weight': 'Weight',
                'description': 'Brief description of the topic',
                'notes': [
                    'The sum of weights for all topics should be 100',
                    'Provide a brief description for each topic'
                ],
                'output_instruction': 'Return only the JSON format result, without any explanations or other text.'
            },
            'japanese': {
                'instruction': '以下の文書を注意深く分析し、4〜6つの主要なトピックを抽出してください。',
                'json_format': '分析結果を次のJSON形式で返してください：',
                'title': 'トピック名',
                'weight': '重み',
                'description': 'トピックの簡単な説明',
                'notes': [
                    'すべてのトピックの重みの合計は100であるべきです',
                    '各トピックに簡単な説明を提供してください'
                ],
                'output_instruction': 'JSON形式の結果のみを返し、説明やその他のテキストを含めないでください。'
            },
            'korean': {
                'instruction': '다음 문서를 주의 깊게 분석하여 4-6개의 주요 주제를 추출하십시오.',
                'json_format': '분석 결과를 다음 JSON 형식으로 반환하십시오:',
                'title': '주제 이름',
                'weight': '가중치',
                'description': '주제에 대한 간략한 설명',
                'notes': [
                    '모든 주제의 가중치 합계는 100이어야 합니다',
                    '각 주제에 대한 간략한 설명을 제공하십시오'
                ],
                'output_instruction': 'JSON 형식 결과만 반환하고 설명이나 다른 텍스트를 포함하지 마십시오.'
            },
            'french': {
                'instruction': 'Analysez attentivement le document suivant, extrayez 4 à 6 sujets principaux.',
                'json_format': 'Retournez les résultats de l\'analyse au format JSON comme suit:',
                'title': 'Nom du sujet',
                'weight': 'Poids',
                'description': 'Brève description du sujet',
                'notes': [
                    'La somme des poids pour tous les sujets doit être 100',
                    'Fournissez une brève description pour chaque sujet'
                ],
                'output_instruction': 'Retournez uniquement le résultat au format JSON, sans explications ni autre texte.'
            },
            'german': {
                'instruction': 'Analysieren Sie das folgende Dokument sorgfältig, extrahieren Sie 4-6 Hauptthemen.',
                'json_format': 'Geben Sie die Analyseergebnisse im folgenden JSON-Format zurück:',
                'title': 'Themenname',
                'weight': 'Gewichtung',
                'description': 'Kurze Beschreibung des Themas',
                'notes': [
                    'Die Summe der Gewichtungen für alle Themen sollte 100 betragen',
                    'Geben Sie für jedes Thema eine kurze Beschreibung an'
                ],
                'output_instruction': 'Geben Sie nur das JSON-Format-Ergebnis zurück, ohne Erklärungen oder anderen Text.'
            },
            'spanish': {
                'instruction': 'Analice cuidadosamente el siguiente documento, extraiga 4-6 temas principales.',
                'json_format': 'Devuelva los resultados del análisis en formato JSON de la siguiente manera:',
                'title': 'Nombre del tema',
                'weight': 'Peso',
                'description': 'Breve descripción del tema',
                'notes': [
                    'La suma de los pesos para todos los temas debe ser 100',
                    'Proporcione una breve descripción para cada tema'
                ],
                'output_instruction': 'Devuelva solo el resultado en formato JSON, sin explicaciones ni otro texto.'
            },
            'russian': {
                'instruction': 'Внимательно проанализируйте следующий документ, выделите 4-6 основных тем.',
                'json_format': 'Верните результаты анализа в формате JSON следующим образом:',
                'title': 'Название темы',
                'weight': 'Вес',
                'description': 'Краткое описание темы',
                'notes': [
                    'Сумма весов для всех тем должна быть 100',
                    'Предоставьте краткое описание для каждой темы'
                ],
                'output_instruction': 'Верните только результат в формате JSON, без пояснений или другого текста.'
            }
        }
        
        # 默认使用中文提示
        language = target_language if target_language in lang_prompts else 'chinese'
        prompt_template = lang_prompts[language]
        
        # 构建提示词
        notes_text = "\n".join([f"- {note}" for note in prompt_template['notes']])
        
        topic_prompt = f"""{prompt_template['instruction']}

文档内容:
{text_for_analysis}

{prompt_template['json_format']}
{{
  "topics": [
    {{
      "title": "{prompt_template['title']}",
      "weight": 35.0,
      "description": "{prompt_template['description']}"
    }}
    // ...
  ]
}}

{notes_text}

{prompt_template['output_instruction']}"""
        
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
                    
                    # 规范化处理结果
                    # 确保权重总和为100
                    total_weight = sum(topic.get('weight', 0) for topic in result['topics'])
                    if total_weight > 0 and abs(total_weight - 100) > 1:
                        # 如果总权重不接近100，则进行归一化
                        for topic in result['topics']:
                            topic['weight'] = round(topic.get('weight', 0) / total_weight * 100, 2)
                    
                    # 确保每个主题都有必要的字段
                    for topic in result['topics']:
                        if 'title' not in topic:
                            topic['title'] = '未命名主题'
                        if 'weight' not in topic:
                            topic['weight'] = 100 / len(result['topics'])
                        if 'description' not in topic:
                            topic['description'] = f"关于{topic['title']}的信息"
                        
                        # 确保权重是浮点数
                        topic['weight'] = float(topic['weight'])
                    
                    # 为每个主题生成关键词
                    for topic in result['topics']:
                        topic_title = topic['title']
                        topic_desc = topic.get('description', '')
                        
                        # 构建主题关键词提取的输入文本
                        topic_text = f"{topic_title}。{topic_desc}"
                        
                        # 使用已有的关键词生成函数生成每个主题的关键词
                        try:
                            print(f"为主题 '{topic_title}' 生成关键词...")
                            keywords = generate_keywords_with_model(topic_text, target_language)
                            # 限制关键词长度，确保每个关键词不超过8个字符
                            topic['keywords'] = [k[:8] for k in keywords[:5]]
                            print(f"已生成关键词: {topic['keywords']}")
                        except Exception as ke:
                            print(f"为主题 '{topic_title}' 生成关键词失败: {str(ke)}")
                            # 使用默认关键词
                            default_kw = get_default_keywords(target_language)
                            topic['keywords'] = default_kw[:3]
                    
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
            return get_default_topics(target_language)
            
    except Exception as e:
        print(f"主题分析错误: {str(e)}")
        return get_default_topics(target_language)

def get_default_topics(target_language=None):
    """返回默认的主题分析结果
    
    Args:
        target_language: 目标语言，如果为None则使用中文
    """
    # 不同语言的默认主题
    default_topics = {
        'chinese': {
        'success': True,
        'topics': [
            {
                "title": "主要内容",
                "weight": 35.0,
                    "keywords": ["主题", "关键点", "核心内容", "要点", "要素"],
                "description": "文档的主要内容和核心论述"
            },
            {
                "title": "技术方面",
                "weight": 30.0,
                    "keywords": ["技术", "方法", "实现", "工具", "流程"],
                "description": "涉及的技术内容和实现方法"
            },
            {
                "title": "应用场景",
                "weight": 20.0,
                    "keywords": ["应用", "场景", "使用", "案例", "实例"],
                "description": "文档描述的应用场景和使用方式"
            },
            {
                "title": "发展趋势",
                "weight": 15.0,
                    "keywords": ["趋势", "展望", "未来", "发展", "方向"],
                "description": "相关领域的发展趋势和未来展望"
            }
        ]
        },
        'english': {
            'success': True,
            'topics': [
                {
                    "title": "Main Content",
                    "weight": 35.0,
                    "keywords": ["Topic", "Key Points", "Core", "Points", "Elements"],
                    "description": "Main content and core arguments of the document"
                },
                {
                    "title": "Technical Aspects",
                    "weight": 30.0,
                    "keywords": ["Technical", "Method", "Implementation", "Tools", "Process"],
                    "description": "Technical content and implementation methods involved"
                },
                {
                    "title": "Application Scenarios",
                    "weight": 20.0,
                    "keywords": ["Application", "Scenario", "Usage", "Cases", "Examples"],
                    "description": "Application scenarios and usage methods described in the document"
                },
                {
                    "title": "Development Trends",
                    "weight": 15.0,
                    "keywords": ["Trends", "Outlook", "Future", "Development", "Direction"],
                    "description": "Development trends and future prospects in related fields"
                }
            ]
        },
        'japanese': {
            'success': True,
            'topics': [
                {
                    "title": "主な内容",
                    "weight": 35.0,
                    "keywords": ["主題", "要点", "核心", "ポイント", "要素"],
                    "description": "文書の主な内容と核心的な議論"
                },
                {
                    "title": "技術的側面",
                    "weight": 30.0,
                    "keywords": ["技術", "方法", "実装", "ツール", "プロセス"],
                    "description": "関連する技術的内容と実装方法"
                },
                {
                    "title": "適用シナリオ",
                    "weight": 20.0,
                    "keywords": ["応用", "シナリオ", "使用法", "事例", "例"],
                    "description": "文書に記述された適用シナリオと使用方法"
                },
                {
                    "title": "発展傾向",
                    "weight": 15.0,
                    "keywords": ["傾向", "展望", "未来", "発展", "方向"],
                    "description": "関連分野の発展傾向と将来の見通し"
                }
            ]
        },
        'korean': {
            'success': True,
            'topics': [
                {
                    "title": "주요 내용",
                    "weight": 35.0,
                    "keywords": ["주제", "핵심", "요점", "포인트", "요소"],
                    "description": "문서의 주요 내용 및 핵심 논의"
                },
                {
                    "title": "기술적 측면",
                    "weight": 30.0,
                    "keywords": ["기술", "방법", "구현", "도구", "과정"],
                    "description": "관련 기술 내용 및 구현 방법"
                },
                {
                    "title": "적용 시나리오",
                    "weight": 20.0,
                    "keywords": ["응용", "시나리오", "사용", "사례", "예시"],
                    "description": "문서에 설명된 응용 시나리오 및 사용 방법"
                },
                {
                    "title": "발전 동향",
                    "weight": 15.0,
                    "keywords": ["동향", "전망", "미래", "발전", "방향"],
                    "description": "관련 분야의 발전 동향 및 미래 전망"
                }
            ]
        },
        'french': {
            'success': True,
            'topics': [
                {
                    "title": "Contenu Principal",
                    "weight": 35.0,
                    "keywords": ["Sujet", "Points Clés", "Essence", "Éléments", "Base"],
                    "description": "Contenu principal et arguments centraux du document"
                },
                {
                    "title": "Aspects Techniques",
                    "weight": 30.0,
                    "keywords": ["Technique", "Méthode", "Mise en œuvre", "Outils", "Processus"],
                    "description": "Contenu technique et méthodes d'implémentation impliquées"
                },
                {
                    "title": "Scénarios d'Application",
                    "weight": 20.0,
                    "keywords": ["Application", "Scénario", "Utilisation", "Cas", "Exemples"],
                    "description": "Scénarios d'application et méthodes d'utilisation décrites dans le document"
                },
                {
                    "title": "Tendances de Développement",
                    "weight": 15.0,
                    "keywords": ["Tendances", "Perspectives", "Futur", "Développement", "Direction"],
                    "description": "Tendances de développement et perspectives futures dans les domaines connexes"
                }
            ]
        },
        'german': {
            'success': True,
            'topics': [
                {
                    "title": "Hauptinhalt",
                    "weight": 35.0,
                    "keywords": ["Thema", "Kernpunkte", "Kern", "Elemente", "Basis"],
                    "description": "Hauptinhalt und zentrale Argumente des Dokuments"
                },
                {
                    "title": "Technische Aspekte",
                    "weight": 30.0,
                    "keywords": ["Technik", "Methode", "Umsetzung", "Werkzeuge", "Prozess"],
                    "description": "Technische Inhalte und Implementierungsmethoden"
                },
                {
                    "title": "Anwendungsszenarien",
                    "weight": 20.0,
                    "keywords": ["Anwendung", "Szenario", "Nutzung", "Fälle", "Beispiele"],
                    "description": "Im Dokument beschriebene Anwendungsszenarien und Nutzungsmethoden"
                },
                {
                    "title": "Entwicklungstrends",
                    "weight": 15.0,
                    "keywords": ["Trends", "Ausblick", "Zukunft", "Entwicklung", "Richtung"],
                    "description": "Entwicklungstrends und Zukunftsaussichten in verwandten Bereichen"
                }
            ]
        },
        'spanish': {
            'success': True,
            'topics': [
                {
                    "title": "Contenido Principal",
                    "weight": 35.0,
                    "keywords": ["Tema", "Puntos Clave", "Núcleo", "Elementos", "Base"],
                    "description": "Contenido principal y argumentos centrales del documento"
                },
                {
                    "title": "Aspectos Técnicos",
                    "weight": 30.0,
                    "keywords": ["Técnica", "Método", "Implementación", "Herramientas", "Proceso"],
                    "description": "Contenido técnico y métodos de implementación involucrados"
                },
                {
                    "title": "Escenarios de Aplicación",
                    "weight": 20.0,
                    "keywords": ["Aplicación", "Escenario", "Uso", "Casos", "Ejemplos"],
                    "description": "Escenarios de aplicación y métodos de uso descritos en el documento"
                },
                {
                    "title": "Tendencias de Desarrollo",
                    "weight": 15.0,
                    "keywords": ["Tendencias", "Perspectivas", "Futuro", "Desarrollo", "Dirección"],
                    "description": "Tendencias de desarrollo y perspectivas futuras en campos relacionados"
                }
            ]
        },
        'russian': {
            'success': True,
            'topics': [
                {
                    "title": "Основное содержание",
                    "weight": 35.0,
                    "keywords": ["Тема", "Ключевые моменты", "Суть", "Элементы", "Основа"],
                    "description": "Основное содержание и центральные аргументы документа"
                },
                {
                    "title": "Технические аспекты",
                    "weight": 30.0,
                    "keywords": ["Техника", "Метод", "Реализация", "Инструменты", "Процесс"],
                    "description": "Технический контент и методы реализации"
                },
                {
                    "title": "Сценарии применения",
                    "weight": 20.0,
                    "keywords": ["Применение", "Сценарий", "Использование", "Примеры", "Случаи"],
                    "description": "Сценарии применения и методы использования, описанные в документе"
                },
                {
                    "title": "Тенденции развития",
                    "weight": 15.0,
                    "keywords": ["Тенденции", "Перспективы", "Будущее", "Развитие", "Направление"],
                    "description": "Тенденции развития и перспективы в смежных областях"
                }
            ]
        }
    }
    
    # 如果没有指定语言或者指定的语言没有对应的默认主题，使用中文
    if not target_language or target_language not in default_topics:
        return default_topics['chinese']
        
    return default_topics[target_language]

def get_embeddings_model():
    """获取统一的嵌入模型 - 使用Ollama的jina-embeddings-v2-base-zh"""
    try:
        
        # 使用Ollama中已安装的jina模型
        embeddings = OllamaEmbeddings(
            model="EntropyYue/jina-embeddings-v2-base-zh",  # 使用您在Ollama中的模型名称
            base_url="http://localhost:11434"  # Ollama默认URL
        )
        
        print("成功加载 Ollama Embeddings: EntropyYue/jina-embeddings-v2-base-zh")
        return embeddings
    except Exception as e:
        print(f"加载 Ollama Embeddings 失败: {str(e)}")
        print("!!! 无法加载嵌入模型，请检查Ollama服务是否运行 !!!")
        raise e  # 抛出异常，因为嵌入模型是核心功能

def create_hybrid_vector_store(text, summary, doc_id):
    """创建混合向量存储"""
    try:
        print("\n=== 开始创建混合向量存储 ===")
        print(f"文档ID: {doc_id}")
        print(f"原始文本长度: {len(text) if text else 0}")
        print(f"摘要文本长度: {len(summary) if summary else 0}")
        
        # 检查RAGTools是否已初始化
        global rag_tools
        if rag_tools is None:
            print("RAGTools尚未初始化，正在尝试初始化...")
            init_rag_tools()
            if rag_tools is None:
                print("RAGTools初始化失败，无法创建向量存储")
                return False
        
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

def hybrid_semantic_search(query, vector_store, content_weight=0.5, summary_weight=0.5, 
                         max_results=5, sliding_window=True, special_terms_boost=True):
    """
    改进的混合语义搜索，结合BM25关键词匹配和向量相似度搜索
    
    参数:
        query: 搜索查询
        vector_store: 向量存储对象
        content_weight: 内容向量的权重
        summary_weight: 摘要向量的权重
        max_results: 返回的最大结果数
        sliding_window: 是否使用滑动窗口
        special_terms_boost: 是否对特殊术语进行增强
    """
    # 1. 提取查询中的专业术语和关键词
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(query)
    
    # 提取名词短语和专业术语
    key_terms = []
    for chunk in doc.noun_chunks:
        key_terms.append(chunk.text)
    
    # 添加更多领域术语检测，扩展专业术语库
    domain_terms = [
        "神经网络", "手语", "识别", "深度学习", "残差网络", "CNN", "卷积神经网络", 
        "深度神经网络", "AI", "人工智能", "机器学习", "计算机视觉", "图像识别", 
        "ResNet", "手势识别", "姿态估计", "手部跟踪", "特征提取", "分类器",
        "手语翻译", "实时识别", "自然语言处理", "NLP", "语义理解", "语义分析"
    ]
    
    # 词语相似度映射，处理同义词和近义词
    term_similarity_map = {
        "残差网络": ["ResNet", "ResidualNet", "res网络"],
        "神经网络": ["深度神经网络", "DNN", "人工神经网络", "ANN"],
        "手语识别": ["手语翻译", "手势识别", "手语理解"],
        "卷积神经网络": ["CNN", "ConvNet", "卷积网络"],
        "计算机视觉": ["CV", "视觉识别", "图像识别"]
    }
    
    # 扩展查询中的关键术语，考虑近义词
    enhanced_terms = []
    for term in key_terms:
        enhanced_terms.append(term)
        # 检查每个关键词是否有相似词
        for base_term, similar_terms in term_similarity_map.items():
            if term in similar_terms or term == base_term:
                # 添加相似词汇
                enhanced_terms.extend([t for t in similar_terms if t != term])
                enhanced_terms.append(base_term)
    
    # 删除重复项
    enhanced_terms = list(set(enhanced_terms))
    print(f"扩展后的查询关键词: {enhanced_terms}")
    
    found_terms = []
    
    # 扩展查询匹配，增加术语识别能力
    for term in domain_terms:
        if term in query:
            found_terms.append(term)
            # 添加该术语的所有相似词
            for base_term, similar_terms in term_similarity_map.items():
                if term == base_term or term in similar_terms:
                    found_terms.extend(similar_terms)
                    found_terms.append(base_term)
    
    # 删除重复项
    found_terms = list(set(found_terms))
    
    # 检测更复杂的领域概念组合
    complex_concepts = {
        "手语识别系统": ["手语", "识别", "系统"],
        "基于残差网络的识别": ["基于", "残差", "网络", "识别"],
        "神经网络手语识别": ["神经", "网络", "手语", "识别"],
        "深度学习手语翻译": ["深度", "学习", "手语", "翻译"]
    }
    
    # 检查是否存在复杂概念的成分词
    for concept, components in complex_concepts.items():
        if all(component in query.lower() for component in components):
            print(f"检测到复杂概念: {concept}")
            found_terms.append(concept)
    
    # 2. BM25关键词匹配，增强对关键术语的重视
    bm25_results = []
    documents = vector_store.get("documents")
    if documents:
        # 为关键术语构建增强的文档表示
        enhanced_docs = []
        for doc in documents:
            # 原始文档
            doc_text = doc
            
            # 计算关键术语在文档中的出现次数
            term_counts = {}
            for term in found_terms + enhanced_terms:
                count = doc_text.lower().count(term.lower())
                if count > 0:
                    term_counts[term] = count
            
            # 根据术语出现次数增强文档表示
            enhanced_doc = doc_text
            for term, count in term_counts.items():
                # 对重要术语进行加权增强
                if count > 0:
                    # 最多重复添加3次以避免过度膨胀
                    repeat_count = min(count, 3)
                    term_addition = f" {term}" * repeat_count
                    enhanced_doc += term_addition
            
            enhanced_docs.append(enhanced_doc)
            
        # 使用增强的文档集进行BM25匹配
        tokenized_corpus = [doc.split() for doc in enhanced_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 将查询分词 - 增加对找到的术语的权重
        enriched_query = " ".join(key_terms + found_terms + enhanced_terms)
        tokenized_query = enriched_query.split()
        print(f"增强的BM25查询: {enriched_query}")
        
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 标准化BM25分数
        if max(bm25_scores) > 0:
            bm25_scores = [score/max(bm25_scores) for score in bm25_scores]
    
    # 3. 向量相似度搜索
    embeddings = generate_embeddings(query)
    vector_results = vector_store.similarity_search_by_vector(
        embeddings, 
        k=max_results*3  # 增加检索数量确保不遗漏相关结果
    )
    
    # 4. 混合得分计算
    final_results = []
    for i, result in enumerate(vector_results):
        if i < len(bm25_scores):
            # 计算向量得分
            content_score = result.content_score if hasattr(result, 'content_score') else 0
            summary_score = result.summary_score if hasattr(result, 'summary_score') else 0
            vector_score = content_weight * content_score + summary_weight * summary_score
            
            # 提取文档文本进行术语匹配分析
            doc_text = result.page_content.lower()
            
            # 混合得分，增加关键术语权重
            term_boost = 0
            matched_terms = []
            
            if special_terms_boost:
                # 检查文档中是否存在我们确定的领域术语
                for term in found_terms + enhanced_terms:
                    term_lower = term.lower()
                    if term_lower in doc_text:
                        # 根据术语重要性给予不同加权
                        if term in found_terms:
                            boost = 0.2  # 直接从查询中提取的术语给予更高权重
                        else:
                            boost = 0.1  # 扩展得到的术语给予较低权重
                        
                        term_boost += boost
                        matched_terms.append(term)
                
                # 复杂概念匹配给予额外奖励
                for concept in complex_concepts:
                    if concept.lower() in doc_text:
                        term_boost += 0.3  # 完整概念匹配给予更高奖励
                        matched_terms.append(concept)
                
                # 手语识别特定领域匹配增强
                if ("手语" in doc_text and "识别" in doc_text) or "手语识别" in doc_text:
                    term_boost += 0.25
                    matched_terms.append("手语识别")
                
                if "残差网络" in doc_text or "ResNet" in doc_text.lower():
                    term_boost += 0.25
                    matched_terms.append("残差网络/ResNet")
            
            # 输出调试信息
            if matched_terms:
                print(f"文档 {i} 匹配到的术语: {matched_terms}, 增强分数: {term_boost}")
            
            # 调整权重分配，提高术语匹配的重要性
            # 最终得分 = 向量得分 * 0.5 + BM25得分 * 0.3 + 术语增强 * 0.2
            final_score = vector_score * 0.5 + bm25_scores[i] * 0.3 + min(term_boost, 1.0) * 0.2
            
            # 限制最大分数为1.0
            final_score = min(final_score, 1.0)
            
            final_results.append({
                "document": result,
                "score": final_score,
                "matched_terms": matched_terms
            })
    
    # 5. 排序并返回结果
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return [item["document"] for item in final_results[:max_results]]

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
                'error': '搜索查询不能为空'
            }), 400
        
        # 查询文档是否存在
        doc = DocumentSummary.query.get(doc_id)
        if not doc:
            return jsonify({
                'success': False,
                'error': '文档不存在'
            }), 404
            
        # 检查文档是否已完成向量化
        if not doc.has_vector_store:
            return jsonify({
                'success': False,
                'error': '文档尚未完成向量化，请稍后再试',
                'pending_vectorization': True
            }), 202  # 返回202 Accepted表示请求已接受但尚未处理完成
            
        # 执行语义搜索
        results = semantic_search(query, doc_id)
        
        if not results:
            return jsonify({
                'success': True,
                'message': '未找到相关内容',
                'results': []
            })
            
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"语义搜索出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}'
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
        max_results = data.get('max_results', 5)
        sliding_window = data.get('sliding_window', True)
        
        if not query:
            return jsonify({
                'success': False,
                'error': '搜索查询不能为空',
                'results': []  # 即使出错也返回空结果数组
            }), 400
            
        # 获取RAGTools实例
        global rag_tools
        if rag_tools is None:
            print("RAGTools未初始化，正在尝试初始化...")
            init_rag_tools()
            if rag_tools is None:
                return jsonify({
                    'success': False,
                    'error': 'RAGTools初始化失败',
                    'results': []
                }), 500
        
        # 获取对应的向量存储
        doc = DocumentSummary.query.get(doc_id)
        if not doc:
            return jsonify({
                'success': False,
                'error': '文档不存在',
                'results': []
            }), 404
            
        # 检查文档是否已完成向量化
        if not doc.has_vector_store:
            return jsonify({
                'success': False,
                'error': '文档尚未完成向量化，请稍后再试',
                'pending_vectorization': True,
                'results': []
            }), 202  # 返回202 Accepted表示请求已接受但尚未处理完成
            
        collection_name = doc.chroma_collection or f"doc_{doc_id}"
        print(f"使用集合名称: {collection_name}")
        
        # 获取Chroma向量存储
        from langchain_chroma import Chroma
        
        try:
            # 指定嵌入模型
            embeddings = get_embeddings_model()
            
            # 确定Chroma集合路径
            persist_directory = os.path.join(os.getcwd(), "chroma_db")
            
            # 加载向量存储
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            
            print(f"成功加载向量存储 {collection_name}")
            
            # 执行混合语义搜索
            results = hybrid_semantic_search(
                query, 
                vector_store, 
                content_weight=content_weight,
                summary_weight=summary_weight,
                max_results=max_results,
                sliding_window=sliding_window
            )
            
            if not results:
                return jsonify({
                    'success': True,
                    'message': '未找到相关内容',
                    'results': []
                })
                
            # 处理结果
            response_results = []
            for i, doc in enumerate(results):
                response_results.append({
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': doc.score if hasattr(doc, 'score') else None,
                    'content_score': doc.content_score if hasattr(doc, 'content_score') else None,
                    'summary_score': doc.summary_score if hasattr(doc, 'summary_score') else None,
                })
                
            return jsonify({
                'success': True,
                'results': response_results
            })
            
        except Exception as e:
            print(f"执行混合语义搜索时出错: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'搜索失败: {str(e)}',
                'results': []
            }), 500
            
    except Exception as e:
        print(f"混合语义搜索请求处理出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}',
            'results': []
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
        # 记录开始时间用于性能统计
        start_time = time.time()
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False, 
                'error': '请求数据不能为空',
                'results': []
            }), 400
            
        query = data.get('query', '').strip()
        use_hybrid = data.get('use_hybrid', True)  # 默认使用混合检索
        min_score_threshold = data.get('min_score', 0.15)  # 进一步降低相关度阈值到0.15
        max_results = data.get('max_results', 20)  # 限制返回结果数量，默认20条
        enable_text_match = data.get('text_match', True)  # 启用文本匹配兜底
        
        if not query:
            return jsonify({
                'success': False, 
                'error': '搜索关键词不能为空',
                'results': []
            }), 400
            
        print(f"\n=== 搜索文档 关键词: '{query}' ===")
        print(f"查询参数: 筛选阈值={min_score_threshold}, 最大结果数={max_results}, 文本匹配启用={enable_text_match}, 混合检索={use_hybrid}")
        
        # 使用预处理函数增强查询，特别针对手语识别和神经网络等领域术语
        enhanced_query = preprocess_search_query(query)
        if enhanced_query != query:
            print(f"查询增强: '{query}' -> '{enhanced_query}'")
        
        # 获取所有文档
        summaries = DocumentSummary.query.all()
        print(f"找到 {len(summaries)} 个文档")
        
        # 将查询转换为小写，便于文本匹配
        query_lower = enhanced_query.lower()
        print(f"搜索关键词(小写): '{query_lower}'")
        
        results = []
        text_match_count = 0
        vector_match_count = 0
        
        # 初始化向量搜索
        embeddings = None
        query_vector = None
        can_use_vector = False
        
        if use_hybrid:
            try:
                embeddings = get_embeddings_model()
                # 使用增强的查询生成向量，提高领域相关性
                query_vector = embeddings.embed_query(enhanced_query)
                can_use_vector = True
                print("成功生成查询向量，可以使用向量搜索")
            except Exception as e:
                print(f"向量模型初始化失败: {str(e)}")
                print("将仅使用文本匹配")
                can_use_vector = False
        
        # 初始化统计数据
        vector_attempts = 0
        vector_success = 0
        text_attempts = 0
        text_success = 0
        all_vector_scores = []
        
        # 遍历所有文档
        for summary in summaries:
            try:
                doc_id = summary.id
                file_name = summary.file_name or ""
                summary_text = summary.summary_text or ""
                keywords = summary.keywords or ""
                
                print(f"\n处理文档: {doc_id} - {file_name}")
                
                # 初始化变量
                found_match = False
                best_match_text = ""
                best_match_score = 0
                best_match_source = ""
                is_text_match = False
                
                # 1. 首先尝试文本匹配
                if enable_text_match:
                    text_attempts += 1
                    
                    # 检查文件名匹配
                    if file_name:
                        # 添加更多调试信息
                        file_name_lower = file_name.lower()
                        print(f"文件名: '{file_name}', 转小写: '{file_name_lower}'")
                        print(f"查询词: '{query}', 增强查询: '{enhanced_query}', 转小写: '{query_lower}'")
                        
                        # 尝试多种匹配方法
                        filename_match = False
                        
                        # 1. 直接包含匹配 - 使用原始查询
                        if query.lower() in file_name_lower:
                            filename_match = True
                            print(f"✓ 文件名直接匹配成功: '{query.lower()}' 在 '{file_name_lower}' 中")
                        # 2. 分词后部分匹配 - 处理文件名中的分隔符
                        elif any(query_lower in part.lower() for part in re.split(r'[_\-\s.]+', file_name)):
                            filename_match = True
                            print(f"✓ 文件名分词匹配成功: '{query_lower}' 匹配了文件名的某一部分")
                        # 3. 尝试Unicode规范化后匹配 - 处理中文编码差异
                        elif unicodedata.normalize('NFKC', query_lower) in unicodedata.normalize('NFKC', file_name_lower):
                            filename_match = True
                            print(f"✓ 文件名Unicode规范化后匹配成功")
                            
                        if filename_match:
                            found_match = True
                            is_text_match = True
                            best_match_text = file_name
                            best_match_score = 0.6  # 文件名匹配给较高分数
                            best_match_source = "file_name"
                            text_match_count += 1
                            text_success += 1
                    
                    # 检查摘要文本匹配
                    elif summary_text and query_lower in summary_text.lower():
                        print(f"✓ 摘要文本匹配: '{query_lower}' 在摘要中")
                        found_match = True
                        is_text_match = True
                        best_match_text = summary_text
                        best_match_score = 0.4  # 摘要匹配给中等分数
                        best_match_source = "summary"
                        text_match_count += 1
                        text_success += 1

                    # 检查关键词匹配
                    elif keywords:
                        # 转换关键词格式
                        if isinstance(keywords, str):
                            keywords_list = keywords.split(',')
                        else:
                            keywords_list = keywords

                        keywords_list = [k.strip().lower() for k in keywords_list if k.strip()]
                        
                        # 添加调试信息
                        print(f"关键词列表: {keywords_list}")
                        print(f"查询词: '{query_lower}'")
                        
                        # 优化匹配逻辑
                        keyword_match = False
                        matched_keyword = ""
                        
                        # 1. 直接匹配
                        if query_lower in keywords_list:
                            keyword_match = True
                            matched_keyword = query_lower
                            print(f"✓ 关键词完全匹配: '{query_lower}' 在关键词列表中")
                        
                        # 2. 部分匹配 - 查询词在某个关键词中
                        elif any(query_lower in k for k in keywords_list):
                            for k in keywords_list:
                                if query_lower in k:
                                    keyword_match = True
                                    matched_keyword = k
                                    print(f"✓ 查询词是关键词的子串: '{query_lower}' 在 '{k}' 中")
                                    break
                        
                        # 3. 部分匹配 - 关键词在查询词中
                        elif any(k in query_lower for k in keywords_list):
                            for k in keywords_list:
                                if k in query_lower:
                                    keyword_match = True
                                    matched_keyword = k
                                    print(f"✓ 关键词是查询词的子串: '{k}' 在 '{query_lower}' 中")
                                    break
                        
                        # 4. Unicode规范化后匹配
                        else:
                            norm_query = unicodedata.normalize('NFKC', query_lower)
                            for k in keywords_list:
                                norm_k = unicodedata.normalize('NFKC', k)
                                if norm_query in norm_k or norm_k in norm_query:
                                    keyword_match = True
                                    matched_keyword = k
                                    print(f"✓ Unicode规范化后关键词匹配成功: '{k}' 与 '{query_lower}'")
                                    break
                        
                        if keyword_match:
                            found_match = True
                            is_text_match = True
                            best_match_text = matched_keyword
                            best_match_score = 0.3  # 关键词匹配给适当分数
                            best_match_source = "keywords"
                            text_match_count += 1
                            text_success += 1
                
                # 2. 如果没有文本匹配，且可以使用向量搜索，进行向量搜索
                if not found_match and can_use_vector:
                    # 首先尝试使用RAGTools
                    if summary.has_vector_store and summary.chroma_collection and rag_tools:
                        vector_attempts += 1
                        
                        try:
                            print(f"使用RAGTools对文档ID {doc_id} 执行向量搜索，top_k=3")
                            vector_results = rag_tools.semantic_search(query, doc_id, top_k=3)
                            
                            if vector_results and len(vector_results) > 0:
                                print(f"向量搜索返回 {len(vector_results)} 个结果")
                                
                                # 输出所有结果的分数，便于调试
                                scores = [result.get('score', 0) for result in vector_results]
                                all_vector_scores.extend(scores)  # 收集所有向量分数
                                
                                if scores:
                                    avg_score = sum(scores) / len(scores)
                                    max_score = max(scores)
                                    min_score = min(scores)
                                    print(f"分数分布: 最高={max_score:.4f}, 最低={min_score:.4f}, 平均={avg_score:.4f}")
                                
                                # 获取最佳匹配结果
                                best_result = max(vector_results, key=lambda x: x.get('score', 0))
                                score = best_result.get('score', 0)
                                source = best_result.get('metadata', {}).get('source', 'content')
                                
                                # 为不同来源应用不同权重
                                score = score * (0.7 if source == 'content' else 0.3)
                                
                                print(f"向量搜索最佳匹配分数: {score:.4f}, 来源: {source}")
                                
                                # 检查分数是否高于阈值
                                if score > min_score_threshold:
                                    found_match = True
                                    is_text_match = False
                                    best_match_text = best_result.get('text', '')
                                    best_match_score = score
                                    best_match_source = source
                                    vector_match_count += 1
                                    vector_success += 1
                                    print(f"✓ 向量搜索匹配成功，分数 {score:.4f} > 阈值 {min_score_threshold}")
                                else:
                                    print(f"✗ 向量搜索分数 {score:.4f} 低于阈值 {min_score_threshold}，忽略此结果")
                                    # 清空向量结果，因为相似度太低
                                    vector_results = []
                            else:
                                print("向量搜索未返回任何结果")
                        except Exception as e:
                            print(f"RAGTools搜索失败: {str(e)}")
                            traceback.print_tb(e.__traceback__ if hasattr(e, "__traceback__") else None)
                            print("继续使用其他方法尝试搜索")
                            
                            # 使用备用方法尝试搜索
                            try:
                                print("尝试使用备用搜索方法...")
                                vector_results = rag_tools.semantic_search_fallback(query, doc_id, top_k=3)
                                
                                if vector_results and len(vector_results) > 0:
                                    print(f"备用方法向量搜索返回 {len(vector_results)} 个结果")
                                    
                                    # 输出所有结果的分数，便于调试
                                    scores = [result.get('score', 0) for result in vector_results]
                                    all_vector_scores.extend(scores)  # 收集所有向量分数
                                    
                                    if scores:
                                        avg_score = sum(scores) / len(scores)
                                        max_score = max(scores)
                                        min_score = min(scores)
                                        print(f"分数分布: 最高={max_score:.4f}, 最低={min_score:.4f}, 平均={avg_score:.4f}")
                                    
                                    # 获取最佳匹配结果
                                    best_result = max(vector_results, key=lambda x: x.get('score', 0))
                                    score = best_result.get('score', 0)
                                    source = best_result.get('metadata', {}).get('source', 'content')
                                    
                                    # 为不同来源应用不同权重
                                    score = score * (0.7 if source == 'content' else 0.3)
                                    
                                    print(f"备用方法向量搜索最佳匹配分数: {score:.4f}, 来源: {source}")
                                    
                                    # 检查分数是否高于阈值
                                    if score > min_score_threshold:
                                        found_match = True
                                        is_text_match = False
                                        best_match_text = best_result.get('text', '')
                                        best_match_score = score
                                        best_match_source = source
                                        vector_match_count += 1
                                        vector_success += 1
                                        print(f"✓ 备用方法向量搜索匹配成功，分数 {score:.4f} > 阈值 {min_score_threshold}")
                                    else:
                                        print(f"✗ 备用方法向量搜索分数 {score:.4f} 低于阈值 {min_score_threshold}，忽略此结果")
                                else:
                                    print("备用方法向量搜索未返回任何结果")
                            except Exception as e2:
                                print(f"备用方法向量搜索也失败: {str(e2)}")
                                print("继续使用传统搜索方法")
                
                # 如果找到了匹配，添加到结果列表
                if found_match:
                    # 使用原始文件名作为显示名称
                    display_name = summary.original_filename or summary.display_filename or summary.file_name
                    
                    # 只显示文件名，不显示路径
                    display_name = os.path.basename(display_name)
                    
                    # 提取匹配文本摘录
                    match_excerpt = best_match_text
                    if len(match_excerpt) > 200:
                        match_excerpt = match_excerpt[:200] + "..."
                    
                    # 处理关键词
                    keywords_array = []
                    if summary.keywords:
                        if isinstance(summary.keywords, str):
                            keywords_array = [k.strip() for k in summary.keywords.split('|') if k.strip()]
                        elif isinstance(summary.keywords, list):
                            keywords_array = summary.keywords
                    
                    # 确保分数至少为0.1（10%），避免显示为0%
                    if best_match_score <= 0:
                        best_match_score = 0.1
                    
                    results.append({
                        'id': summary.id,
                        'file_name': display_name,
                        'summary_text': summary.summary_text,
                        'best_match_text': best_match_text,
                        'match_excerpt': match_excerpt,
                        'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'target_language': summary.target_language,
                        'summary_length': summary.summary_length,
                        'score': float(best_match_score),
                        'relevance_score': float(best_match_score),  # 添加relevance_score字段，确保与前端兼容
                        'keywords': summary.keywords.split('|') if summary.keywords else [],
                        'topic_analysis': summary.topic_analysis,
                        'match_source': best_match_source,
                        'is_text_match': is_text_match
                    })
            
                    print(f"✓ 添加到结果: ID={summary.id}, 分数={best_match_score:.4f}, 匹配源={best_match_source}, 是文本匹配={is_text_match}")
                else:
                    print(f"✗ 未找到匹配")
            
            except Exception as e:
                print(f"处理文档 {summary.id} 时出错: {str(e)}")
                traceback.print_exc()
                continue
        
        # 按相关度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 限制结果数量
        if max_results > 0 and len(results) > max_results:
            results = results[:max_results]
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 计算向量分数统计（如果有）
        vector_score_stats = {}
        if all_vector_scores:
            vector_score_stats = {
                'count': len(all_vector_scores),
                'max': max(all_vector_scores),
                'min': min(all_vector_scores),
                'avg': sum(all_vector_scores) / len(all_vector_scores),
                'above_threshold': len([s for s in all_vector_scores if s > min_score_threshold]),
                'threshold_rate': len([s for s in all_vector_scores if s > min_score_threshold]) / len(all_vector_scores)
            }

        # 计算成功率
        text_success_rate = text_success / text_attempts if text_attempts > 0 else 0
        vector_success_rate = vector_success / vector_attempts if vector_attempts > 0 else 0

        # 输出统计信息
        total_matches = text_match_count + vector_match_count
        print(f"\n=== 搜索结果统计 ===")
        print(f"查询: '{query}'")
        print(f"总文档数: {len(summaries)}")
        print(f"共找到 {total_matches} 个相关文档 (文本匹配: {text_match_count}, 向量匹配: {vector_match_count})")
        print(f"文本匹配尝试: {text_attempts}, 成功: {text_success}, 成功率: {text_success_rate:.2%}")
        print(f"向量匹配尝试: {vector_attempts}, 成功: {vector_success}, 成功率: {vector_success_rate:.2%}")

        if vector_score_stats:
            print(f"向量分数统计: 最高={vector_score_stats['max']:.4f}, 最低={vector_score_stats['min']:.4f}, 平均={vector_score_stats['avg']:.4f}")
            print(f"超过阈值向量匹配比率: {vector_score_stats['threshold_rate']:.2%} ({vector_score_stats['above_threshold']}/{vector_score_stats['count']})")

        print(f"返回 {len(results)} 条结果")
        print(f"搜索执行时间: {execution_time:.4f}秒")
        print(f"=== 搜索完成 ===\n")

        # 添加统计信息到返回结果
        statistics = {
            'execution_time': execution_time,
            'total_docs': len(summaries),
            'text_attempts': text_attempts,
            'text_success': text_success,
            'text_success_rate': text_success_rate,
            'vector_attempts': vector_attempts,
            'vector_success': vector_success,
            'vector_success_rate': vector_success_rate,
            'vector_score_stats': vector_score_stats
        }
            
        return jsonify({
            'success': True,
            'results': results,
            'query': query,
            'result_count': len(results),
            'vector_match': vector_match_count,
            'text_match': text_match_count,
            'execution_time': execution_time,
            'statistics': statistics,
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
            print(f"使用现有的主题分析数据，文档ID: {summary_id}")
            return jsonify(doc.topic_analysis)
        
        # 没有分析数据，使用文档内容进行分析
        if not doc.original_text:
            return jsonify({
                'success': False,
                'error': '文档内容为空，无法进行主题分析'
            }), 400
            
        # 获取目标语言
        target_language = doc.target_language
        print(f"开始为文档(ID: {summary_id})进行主题分析，目标语言: {target_language}")
        
        # 执行主题分析，明确传递目标语言参数
        analysis_result = analyze_document_topics(doc.original_text, target_language)
        
        if not analysis_result or not analysis_result.get('success', False):
            print(f"主题分析失败，使用默认主题，文档ID: {summary_id}")
            analysis_result = get_default_topics(target_language)
        
        # 确保结果格式正确
        if 'topics' not in analysis_result:
            print(f"分析结果缺少topics字段，使用默认主题，文档ID: {summary_id}")
            analysis_result = get_default_topics(target_language)
        
        # 保存分析结果到数据库
        doc.topic_analysis = analysis_result
        db.session.commit()
        print(f"主题分析完成并保存到数据库，文档ID: {summary_id}")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"主题分析API错误: {str(e)}")
        traceback.print_exc()
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
@login_required
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
        
        # 如果从模型回复中提取的关键词不足，使用generate_keywords_with_model函数生成与目标语言匹配的关键词
        if len(keywords_list) < 3:
            print(f"从模型回复中提取的关键词不足，使用generate_keywords_with_model函数生成{target_language}语言的关键词")
            keywords_list = generate_keywords_with_model(summary_text, target_language)
        
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
                # 调用保存摘要函数
                save_summary_to_db(file_info, summary_text, params, file_content)
                print("摘要已成功保存到数据库")
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
        
            # 使用存储的原始文本
            total_pages = (len(text_content) + 5000 - 1) // 5000
            initial_content = text_content[:5000]
            
            # 特殊处理markdown文件，确保代码块的完整性
            if file_type == 'md' and '```' in initial_content:
                # 检查最后一个代码块是否未闭合
                code_blocks = initial_content.split('```')
                if len(code_blocks) % 2 == 0:  # 偶数表示代码块未闭合
                    # 尝试找到下一个闭合标记
                    next_block_start = text_content.find('```', 5000)
                    if next_block_start != -1:
                        # 找到下一个闭合，扩展initial_content到这个位置后
                        next_block_end = text_content.find('\n', next_block_start)
                        if next_block_end != -1:
                            initial_content = text_content[:next_block_end+1]
        
        # 处理markdown文档，转义HTML特殊字符
        if file_type == 'md':
            initial_content = initial_content.replace('<', '&lt;').replace('>', '&gt;')
        
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
        # 使用本地Ollama嵌入模型
        try:
            # 检查GPU是否可用
            import torch
            use_gpu = torch.cuda.is_available()
            device = 'cuda' if use_gpu else 'cpu'
            
            # 使用全局的embeddings模型函数获取模型
            self.embeddings = get_embeddings_model()
            print(f"RAGTools初始化成功，使用嵌入模型: {self.embeddings.model if hasattr(self.embeddings, 'model') else 'unknown'}")
        except Exception as e:
            print(f"初始化过程出现错误: {str(e)}")
            # 回退到基础Ollama嵌入模型
            self.embeddings = OllamaEmbeddings(
                model="snowflake-arctic-embed2",
                base_url="http://localhost:11434"
            )
            print("使用基础Ollama嵌入模型")
            
        # 优化：增加chunk_size和chunk_overlap以提高语义连贯性
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,  # 增加块大小以包含更多上下文信息
            chunk_overlap=450,  # 25%的重叠率以确保概念连续性
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.persist_directory = "chroma_db"
        
    def create_vector_store(self, texts, doc_id, metadata=None):
        """创建向量存储，用于文档检索"""
        try:
            print(f"==== 开始创建向量存储，文档ID: {doc_id} ====")
            # 检查输入是否为空
            if not texts or all(not text.strip() for text in texts):
                print("错误: 输入文本为空或只包含空白字符")
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
                        'source': 'content',
                        'type': 'original'  # 添加类型标记
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
                traceback.print_exc()
                return False
            
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
                traceback.print_exc()
                return False
            
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
        """语义搜索增强版"""
        try:
            # 优化查询：处理特殊术语
            enhanced_query = query
            special_terms = {"图书管理系统": 3, "文档管理": 2, "数据库": 2}
            for term, count in special_terms.items():
                if term in query:
                    # 重复重要术语以增强其权重
                    term_addition = f" {term} " * count
                    enhanced_query = f"{query} {term_addition}"
                    print(f"查询增强: 添加了 {count}x '{term}'")
            
            # 获取向量存储
            collection_name = f"doc_{doc_id}"
            try:
                db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                # 创建检索器 - 请求比所需更多的结果以便后处理
                retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k * 2} # 获取2倍于所需结果数量的候选项
                )
                
                # 执行检索
                print(f"执行向量搜索，查询：'{enhanced_query}'，文档ID：{doc_id}，初始top_k：{top_k*2}")
                docs = retriever.get_relevant_documents(enhanced_query)
                print(f"检索到 {len(docs)} 个相关文档")
            except Exception as e:
                print(f"获取Chroma集合或执行检索时出错: {str(e)}")
                traceback.print_exc()
                return []
            
            # 计算查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 格式化结果并计算相似度分数
            results = []
            for i, doc in enumerate(docs):
                # 计算文档内容与查询的相似度分数
                try:
                    # 初始默认分数
                    similarity_score = 0.0
                    
                    # 计算余弦相似度
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_score = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
                    
                    # 确保分数在0-1范围内
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    
                    # 获取元数据
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    results.append({
                        'text': doc.page_content,
                        'metadata': metadata,
                        'score': similarity_score
                    })
                    
                    print(f"文档 {i+1}: 相似度分数 = {similarity_score:.4f}")
                    
                except Exception as e:
                    print(f"计算文档 {i+1} 相似度时出错: {str(e)}")
                    # 仍然添加到结果中，但分数为0
                    results.append({
                        'text': doc.page_content,
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                        'score': 0.0
                    })
            
            # 按相似度分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # 应用滑动窗口优化: 检测连续文本块并提升其权重
            enhanced_results = results.copy()
            
            # 确定块的上下文窗口
            context_window = []
            for result in enhanced_results[:top_k]:
                chunk_id = result['metadata'].get('chunk_id')
                chunk_index = result['metadata'].get('chunk_index')
                if chunk_id is not None and chunk_index is not None:
                    context_window.append((chunk_id, chunk_index))
            
            # 查找附近的块
            nearby_chunks = []
            for chunk_id, chunk_index in context_window:
                # 寻找前后相邻的块
                nearby_chunks.extend([
                    (chunk_id, chunk_index - 1),
                    (chunk_id, chunk_index + 1)
                ])
            
            # 对结果应用滑动窗口提升
            for i, result in enumerate(enhanced_results):
                chunk_id = result['metadata'].get('chunk_id')
                chunk_index = result['metadata'].get('chunk_index')
                chunk_tuple = (chunk_id, chunk_index)
                
                # 如果此块在我们的相邻窗口中，给予额外权重
                if chunk_tuple in nearby_chunks:
                    old_score = result['score']
                    result['score'] = min(1.0, old_score * 1.25)  # 25%的提升
                    print(f"滑动窗口提升: 块 ({chunk_id}, {chunk_index}) 分数从 {old_score:.4f} 到 {result['score']:.4f}")
            
            # 重新排序
            enhanced_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 确保返回的结果数不超过请求的top_k
            return enhanced_results[:top_k]
            
        except Exception as e:
            print(f"语义搜索失败: {str(e)}")
            traceback.print_exc()
            return []
            
    def generate_rag_response(self, query, doc_id, streaming=False):
        """生成RAG响应"""
        try:
            # 获取向量存储
            collection_name = f"doc_{doc_id}"
            try:
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
            except Exception as e:
                print(f"获取Chroma集合或创建检索器时出错: {str(e)}")
                traceback.print_exc()
                return None
            
            # 获取文档的目标语言
            doc = DocumentSummary.query.get(doc_id)
            target_language = "chinese"  # 默认中文
            if doc and doc.target_language:
                target_language = doc.target_language
            
            print(f"使用目标语言: {target_language}")
            
            # 创建LLM
            llm = Ollama(
                model="huihui_ai/qwen2.5-1m-abliterated",
                base_url="http://localhost:11434",
                streaming=streaming,
                callbacks=[StreamingStdOutCallbackHandler()] if streaming else None
            )
            
            # 获取语言相关的CoT提示词
            language_prompt = generate_cot_language_prompt(target_language, "回答")
            
            # 创建提示模板，使用CoT方法确保语言一致性
            template = f"""使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文: {{context}}

问题: {{question}}

{language_prompt}

答案："""
    
            # 语言映射
            language_mapping = {
                "chinese": "中文",
                "english": "英文",
                "japanese": "日文",
                "korean": "韩文",
                "french": "法文",
                "german": "德文",
                "spanish": "西班牙文",
                "russian": "俄文"
            }
            
            language_text = language_mapping.get(target_language, "中文")
            
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )
            
            # 创建问答链
            # 使用新版langchain API
            stuff_documents_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
            qa_chain = create_retrieval_chain(retriever, stuff_documents_chain)
            
            # 执行问答
            response = qa_chain.invoke({"input": query})
            
            return {
                'answer': response["answer"],
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in response.get('context', [])
                ]
            }
            
        except Exception as e:
            print(f"生成RAG响应失败: {str(e)}")
            traceback.print_exc()
            return None

@app.route('/test_keywords')
def test_keywords():
    """测试路由：检查数据库中的关键词格式"""
    try:
        # 获取所有摘要记录
        results = DocumentSummary.query.filter(DocumentSummary.keywords.isnot(None)).limit(10).all()
        
        # 收集数据
        summaries_data = []
        for summary in results:
            # 提取原始关键词数据
            raw_keywords = summary.keywords
            
            # 处理关键词 - 同get_summaries函数中的逻辑
            keywords_array = []
            if raw_keywords:
                if isinstance(raw_keywords, str) and raw_keywords.strip():
                    if '|' in raw_keywords:
                        keywords_array = [k.strip() for k in raw_keywords.split('|') if k.strip()]
                    else:
                        keywords_array = [raw_keywords.strip()]
                elif isinstance(raw_keywords, list):
                    keywords_array = [k for k in raw_keywords if k]
                else:
                    # 尝试强制转换
                    try:
                        str_val = str(raw_keywords)
                        keywords_array = [str_val.strip()]
                    except:
                        pass
            
            # 构建结果数据
            data = {
                'id': summary.id,
                'file_name': summary.file_name,
                'raw_keywords': raw_keywords,
                'raw_keywords_type': str(type(raw_keywords)),
                'processed_keywords': keywords_array,
                'processed_keywords_type': str(type(keywords_array))
            }
            summaries_data.append(data)
        
        return jsonify({
            'count': len(summaries_data),
            'summaries': summaries_data,
            'debug_info': {
                'python_version': sys.version,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': None
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'debug_info': {
                'python_version': sys.version,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 500

@app.route('/debug_keywords')
def debug_keywords_page():
    """显示关键词调试页面"""
    return send_from_directory('.', 'debug_data.html')

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
        # The case where [KEYWORDS] exists but [SUMMARY] doesn't
        if keywords_match:
            remaining_text = response[keywords_match.end():].strip()
            if remaining_text:
                print("使用关键词后的内容作为摘要")
                summary_text = remaining_text
        
        # If no valid summary is found, use the entire response
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

def generate_keywords_with_model(text, target_language=None):
    """使用大模型结合Chain-of-Thought技术为文本生成高质量关键词
    
    Args:
        text: 要分析的文本内容
        target_language: 目标语言，如果为None则使用中文
    """
    try:
        if not text:
            print("无法生成关键词：输入文本为空")
            return get_default_keywords(target_language)
            
        print(f"正在使用增强CoT技术生成高质量{target_language or '中文'}关键词，文本长度: {len(text)}")
        
        # 创建Ollama客户端
        client = Client(host='http://localhost:11434')
        
        # 限制文本长度，兼顾效率和准确性
        text_sample = text[:3000] if len(text) > 3000 else text
        
        # 根据目标语言设置不同的提示词
        keyword_prompts = {
            'chinese': """请分析下面的文本，提取4-6个最能代表文本核心内容的中文关键词""",
            'english': """Analyze the text below and extract 4-6 keywords in English that best represent the core content""",
            'japanese': """以下のテキストを分析し、テキストの核心的な内容を最もよく表す4〜6個の日本語キーワードを抽出してください""",
            'korean': """아래 텍스트를 분석하고 핵심 내용을 가장 잘 나타내는 4-6개의 한국어 키워드를 추출하세요""",
            'french': """Analysez le texte ci-dessous et extrayez 4 à 6 mots-clés en français qui représentent le mieux le contenu principal""",
            'german': """Analysieren Sie den untenstehenden Text und extrahieren Sie 4-6 deutsche Schlüsselwörter, die den Kerninhalt am besten repräsentieren""",
            'spanish': """Analice el texto a continuación y extraiga 4-6 palabras clave en español que mejor representen el contenido principal""",
            'russian': """Проанализируйте текст ниже и извлеките 4-6 ключевых слов на русском языке, которые лучше всего представляют основное содержание"""
        }
        
        # 语言特性提示
        language_features = {
            'chinese': "中文关键词通常为2-4个汉字，应该是名词或名词短语",
            'english': "English keywords are typically nouns or noun phrases, often 1-3 words in length",
            'japanese': "日本語のキーワードは通常、名詞または名詞句であり、2〜4文字の漢字またはひらがな/カタカナの組み合わせです",
            'korean': "한국어 키워드는 일반적으로 명사 또는 명사구이며, 2-4자의 한글 또는 한자로 구성됩니다",
            'french': "Les mots-clés français sont généralement des noms ou des groupes nominaux, souvent de 1 à 3 mots",
            'german': "Deutsche Schlüsselwörter sind typischerweise Substantive oder Nominalphrasen, oft mit 1-3 Wörtern",
            'spanish': "Las palabras clave en español suelen ser sustantivos o frases nominales, a menudo de 1 a 3 palabras",
            'russian': "Русские ключевые слова обычно являются существительными или именными словосочетаниями, часто длиной от 1 до 3 слов"
        }
        
        # 默认使用中文提示
        language = target_language if target_language in keyword_prompts else 'chinese'
        prompt_prefix = keyword_prompts[language]
        language_feature = language_features.get(language, language_features['chinese'])
        
        # 使用增强型CoT提示词，强化语言约束
        new_keyword_prompt = f"""{prompt_prefix}：

[文本内容]
{text_sample}

[任务要求]
请生成4-6个{language}关键词，这些关键词必须能够准确代表文本的核心内容。
{language_feature}。

[思维链分析]
1. 分析文本主题和核心概念
2. 识别文本中反复出现的重要术语
3. 归纳文本的关键主题和中心思想
4. 确保所有关键词都使用{language}，符合{language}的语言习惯
5. 检查关键词是否能准确概括文本内容
6. 确保关键词具有专业性和准确性

[输出格式]
只需直接输出用竖线(|)分隔的关键词，不要包含任何解释或额外内容。
例如: 关键词1|关键词2|关键词3|关键词4

[语言要求]
务必确保所有关键词都是纯{language}，不要混合使用其他语言。

现在，直接输出{language}关键词（用竖线分隔）："""
        
        # 调用模型生成关键词，降低温度确保更精确的输出
        keyword_response = client.generate(
            model='huihui_ai/qwen2.5-1m-abliterated:latest',
            prompt=new_keyword_prompt,
            stream=False,
            options={
                'temperature': 0.2, 
                'max_tokens': 100,  # 限制输出长度
                'top_p': 0.85       # 减少随机性
            }
        )
        
        if not keyword_response or 'response' not in keyword_response:
            print("关键词API响应为空或格式错误")
            return get_default_keywords(target_language)
        
        # 提取响应
        response_text = keyword_response['response'].strip()
        print(f"模型生成的原始关键词响应: {response_text}")
        
        # 严格清理和提取关键词
        # 1. 只保留文本中第一行包含分隔符的内容
        first_line_with_separator = None
        for line in response_text.split('\n'):
            if '|' in line:
                first_line_with_separator = line.strip()
                break
        
        if first_line_with_separator:
            # 使用第一行包含分隔符的内容
            keywords_text = first_line_with_separator
        else:
            # 如果没有包含分隔符的行，使用第一行或整个响应
            keywords_text = response_text.split('\n')[0] if '\n' in response_text else response_text
        
        # 2. 拆分关键词
        if '|' in keywords_text:
            raw_keywords = [k.strip() for k in keywords_text.split('|') if k.strip()]
        else:
            # 如果没有分隔符，按空格分词
            raw_keywords = keywords_text.split()
        
        # 3. 验证每个关键词，确保至少有一些关键词
        if not raw_keywords or len(raw_keywords) < 2:
            # 如果没有有效关键词或太少，返回默认关键词
            return get_default_keywords(target_language)
        
        # 4. 确保最终结果不会太长，以防溢出数据库字段
        final_keywords = raw_keywords[:6]  # 最多6个关键词
        
        # 5. 如果关键词生成质量不佳，可以尝试二次验证
        if len(final_keywords) < 3 or any(len(k) > 15 for k in final_keywords):
            # 如果关键词过少或过长，可能质量不佳，进行二次验证
            print(f"关键词质量不佳，尝试使用默认关键词")
            return get_default_keywords(target_language)
        
        # 6. 打印和返回结果
        print(f"最终清理后的{language}关键词: {final_keywords}")
        return final_keywords
            
    except Exception as e:
        print(f"生成关键词失败: {str(e)}")
        traceback.print_exc()
        # 返回默认关键词
        return get_default_keywords(target_language)

def get_default_keywords(target_language):
    """根据目标语言返回默认关键词"""
    default_keywords = {
        'chinese': ["文档摘要", "系统设计", "模型应用", "智能处理"],
        'english': ["Document Summary", "System Design", "Model Application", "Intelligent Processing"],
        'japanese': ["文書要約", "システム設計", "モデル応用", "インテリジェント処理"],
        'korean': ["문서 요약", "시스템 설계", "모델 응용", "지능 처리"],
        'french': ["Résumé Document", "Conception Système", "Application Modèle", "Traitement Intelligent"],
        'german': ["Dokumentzusammenfassung", "Systemdesign", "Modellanwendung", "Intelligente Verarbeitung"],
        'spanish': ["Resumen Documento", "Diseño Sistema", "Aplicación Modelo", "Procesamiento Inteligente"],
        'russian': ["Резюме Документа", "Проектирование Системы", "Применение Модели", "Интеллектуальная Обработка"]
    }
    return default_keywords.get(target_language, default_keywords['chinese'])

# 初始化 RAGTools
rag_tools = None

# 将在应用启动后初始化
def init_rag_tools():
    global rag_tools
    try:
        # 初始化全局RAGTools实例
        rag_tools = RAGTools()
        print("RAGTools 初始化成功")
    except Exception as e:
        print(f"RAGTools 初始化失败: {str(e)}")
        traceback.print_exc()

@app.route('/search')
def search_redirect():
    """兼容旧版前端的搜索路由，将GET请求转发到POST /api/search"""
    
    query = request.args.get('q', '').strip()
    if not query:
        # 如果没有查询参数，返回空结果
        return jsonify([])
        
    print(f"\n=== 搜索重定向: 查询 '{query}' ===")
    
    # 获取所有文档
    summaries = DocumentSummary.query.all()
    
    # 将查询转换为小写，便于文本匹配
    query_lower = query.lower()
    
    results = []
    
    for summary in summaries:
        try:
            file_name = summary.file_name
            summary_text = summary.summary_text
            keywords = summary.keywords
            
            found_match = False
            is_text_match = False
            best_match_text = ""
            best_match_score = 0
            best_match_source = ""
            
            # 检查文件名匹配
            if file_name and query_lower in file_name.lower():
                found_match = True
                is_text_match = True
                best_match_text = file_name
                best_match_score = 0.8
                best_match_source = "file_name"
                
            # 检查摘要文本匹配
            elif summary_text and query_lower in summary_text.lower():
                found_match = True
                is_text_match = True
                best_match_text = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
                best_match_score = 0.6
                best_match_source = "summary_text"
                
            # 检查关键词匹配
            elif keywords:
                # 转换关键词格式
                if isinstance(keywords, str):
                    keywords_list = keywords.split('|')
                else:
                    keywords_list = keywords

                keywords_list = [k.strip().lower() for k in keywords_list if k.strip()]
                
                # 简化匹配逻辑
                if any(query_lower in k or k in query_lower for k in keywords_list):
                    found_match = True
                    is_text_match = True
                    best_match_text = query
                    best_match_score = 0.7
                    best_match_source = "keywords"
            
            # 如果找到了匹配，添加到结果列表
            if found_match:
                # 使用原始文件名作为显示名称
                display_name = summary.original_filename or summary.display_filename or summary.file_name
                
                # 只显示文件名，不显示路径
                display_name = os.path.basename(display_name)
                
                # 提取匹配文本摘录
                match_excerpt = best_match_text
                if len(match_excerpt) > 200:
                    match_excerpt = match_excerpt[:200] + "..."
                
                # 处理关键词
                keywords_array = []
                if summary.keywords:
                    if isinstance(summary.keywords, str):
                        keywords_array = [k.strip() for k in summary.keywords.split('|') if k.strip()]
                    elif isinstance(summary.keywords, list):
                        keywords_array = summary.keywords
                
                # 确保分数至少为0.1（10%），避免显示为0%
                if best_match_score <= 0:
                    best_match_score = 0.1
                        
                results.append({
                    'id': summary.id,
                    'file_name': display_name,
                    'summary_text': summary.summary_text,
                    'match_excerpt': match_excerpt,
                    'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'target_language': summary.target_language,
                    'summary_length': summary.summary_length,
                    'relevance_score': float(best_match_score),
                    'score': float(best_match_score),  # 添加score字段，确保与前端兼容
                    'keywords': keywords_array,
                    'match_source': best_match_source,
                    'is_text_match': is_text_match
                })
        
        except Exception as e:
            print(f"处理文档 {summary.id} 时出错: {str(e)}")
            continue
    
    # 按相关度排序
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # 限制返回结果数量
    results = results[:20]
    
    return jsonify(results)

@app.route('/api/register', methods=['POST'])
def register():
    """注册新用户API"""
    try:
        # 获取JSON数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
            
        # 提取用户信息
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # 验证必填字段
        if not all([username, email, password]):
            return jsonify({"error": "用户名、邮箱和密码为必填项"}), 400
            
        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({"error": "用户名已被使用"}), 409
            
        # 检查邮箱是否已存在
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return jsonify({"error": "邮箱已被注册"}), 409
            
        # 验证用户名不能为admin
        if username.lower() == 'admin':
            return jsonify({"error": "不能使用'admin'作为用户名"}), 400
            
        # 创建新用户
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            role='user'
        )
        
        # 保存到数据库
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({"message": "注册成功", "username": username}), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"用户注册错误: {str(e)}")
        return jsonify({"error": "注册过程中发生错误"}), 500

@app.route('/api/reindex_all', methods=['POST'])
@login_required
def reindex_all_documents():
    """重新对所有文档进行向量化"""
    try:
        # 1. 删除现有的Chroma集合
        import shutil
        import os
        
        chroma_dir = os.path.join(os.getcwd(), 'chroma_db')
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            print(f"已删除现有向量存储目录: {chroma_dir}")
        
        # 2. 重新初始化RAGTools，确保使用新的embedding模型
        init_rag_tools()
        
        # 3. 获取当前用户的所有文档
        user_id = session.get('user_id')
        documents = DocumentSummary.query.filter_by(user_id=user_id).all()
        
        # 4. 重新为每个文档创建向量存储
        total_docs = len(documents)
        processed_docs = 0
        
        for doc in documents:
            try:
                if doc.original_text and doc.summary_text:
                    # 重新创建向量存储
                    create_hybrid_vector_store(doc.original_text, doc.summary_text, doc.id)
                    # 更新文档标记
                    doc.has_vector_store = True
                    # 记录使用的embedding模型
                    doc.embedding_model = "EntropyYue/jina-embeddings-v2-base-zh"
                    processed_docs += 1
                    print(f"已重新索引文档 {doc.id}: {doc.file_name}")
                else:
                    print(f"跳过文档 {doc.id}: {doc.file_name} - 缺少原文或摘要")
            except Exception as e:
                print(f"处理文档 {doc.id} 时出错: {str(e)}")
        
        # 提交所有更改
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'成功重新索引 {processed_docs}/{total_docs} 个文档',
            'processed': processed_docs,
            'total': total_docs
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'重新索引过程中发生错误: {str(e)}'
        }), 500

# 添加一个路由，用于重新初始化RAGTools（而不重启服务器）
@app.route('/api/reinit_rag_tools', methods=['POST'])
@login_required
def reinit_rag_tools_api():
    """重新初始化RAGTools"""
    try:
        init_rag_tools()
        return jsonify({'success': True, 'message': 'RAGTools 重新初始化成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'RAGTools 重新初始化失败: {str(e)}'})

def chunk_text(text, chunk_size=3000, chunk_overlap=600):
    """
    将文本分成重叠的块
    优化版：增加默认块大小和重叠率，并使用更智能的分割策略
    """
    try:
        if not text or not isinstance(text, str):
            return []
            
        # 动态调整块大小和重叠
        text_length = len(text)
        
        # 对于大文档，可以使用更大的块大小，但有上限
        if text_length > 100000:  # 超过10万字符的大文档
            chunk_size = min(4000, chunk_size)
            chunk_overlap = min(800, int(chunk_size * 0.2))  # 20%的重叠率
        elif text_length < 10000:  # 小文档保持较小的块以确保精度
            chunk_size = max(1500, chunk_size)
            chunk_overlap = max(300, int(chunk_size * 0.2))
            
        # 检测文档类型：如果包含大量Markdown标记或结构化文本，使用不同的处理策略
        has_markdown = '##' in text or '**' in text or '*' in text
        has_paragraphs = text.count('\n\n') > text_length / 1000  # 段落密度检测
        
        # 使用更智能的分割策略
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # 定义分隔符，按优先级排列
        separators = [
            "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # Markdown 标题
            "\n\n", "\n",                                          # 段落和换行
            ". ", "! ", "? ",                                      # 句子结束
            "；", "。", "，", "：", "；",                           # 中文标点
            " ", ""                                                # 空格和无分隔符
        ]
        
        # 如果检测到Markdown，优先使用Markdown分隔符
        if has_markdown:
            separators = [
                "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # Markdown 标题
                "\n- ", "\n* ", "\n1. ", "\n> ",                       # Markdown 列表和引用
                "\n\n", "\n",                                          # 段落和换行
                ". ", "! ", "? ",                                      # 句子结束
                "；", "。", "，", "：", "；",                           # 中文标点
                " ", ""                                                # 空格和无分隔符
            ]
        # 如果是段落型文本，优先按段落分割
        elif has_paragraphs:
            separators = [
                "\n\n", "\n",                                          # 段落优先
                ". ", "! ", "? ",                                      # 句子结束
                "；", "。", "，", "：", "；",                           # 中文标点
                " ", ""                                                # 空格和无分隔符
            ]
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # 分割文本
        chunks = text_splitter.create_documents([text])
        
        # 格式化块
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            # 确保块不为空并且不只包含空白字符
            if chunk_text and chunk_text.strip():
                # 计算块中的句子数量（粗略估计）
                sentence_count = sum(1 for c in ".!?。？！" if c in chunk_text)
                
                # 生成文本预览（前50个字符）
                preview = chunk_text[:50] + ("..." if len(chunk_text) > 50 else "")
                
                formatted_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': f"chunk_{uuid.uuid4()}",
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'preview': preview,
                        'sentence_count': sentence_count
                    }
                })
        
        print(f"分块完成：将 {len(text)} 字符分为 {len(formatted_chunks)} 个块，块大小={chunk_size}，重叠={chunk_overlap}")
        return formatted_chunks
        
    except Exception as e:
        error_msg = f"分块文本出错: {str(e)}"
        traceback.print_exc()
        print(error_msg)
        return []

def generate_cot_language_prompt(target_language, content_type="摘要"):
    """
    生成思维链提示词，确保输出内容始终使用指定的语言
    
    参数:
        target_language: 目标语言，如 'chinese', 'english' 等
        content_type: 内容类型，默认为 "摘要"
    
    返回:
        经过优化的CoT提示词
    """
    # 语言映射表
    language_mapping = {
        "chinese": {
            "name": "中文",
            "instruction": "请使用标准中文输出，确保专业术语、标点符号和表达方式符合中文习惯",
            "examples": ["人工智能", "机器学习", "语言模型", "深度学习"]
        },
        "english": {
            "name": "英文",
            "instruction": "请使用专业英文输出，确保术语准确、表达地道",
            "examples": ["artificial intelligence", "machine learning", "language model", "deep learning"]
        },
        "japanese": {
            "name": "日文",
            "instruction": "请使用标准日语输出，注意日语特有的表达方式和敬语",
            "examples": ["人工知能", "機械学習", "言語モデル", "深層学習"]
        },
        "korean": {
            "name": "韩文",
            "instruction": "请使用标准韩语输出，注意韩语特有的语法和表达",
            "examples": ["인공지능", "기계학습", "언어모델", "심층학습"]
        },
        "french": {
            "name": "法文",
            "instruction": "请使用专业法语输出，确保语法和表达符合法语规范",
            "examples": ["intelligence artificielle", "apprentissage automatique", "modèle de langage", "apprentissage profond"]
        },
        "german": {
            "name": "德文",
            "instruction": "请使用专业德语输出，确保语法和表达符合德语规范",
            "examples": ["künstliche Intelligenz", "maschinelles Lernen", "Sprachmodell", "tiefes Lernen"]
        },
        "spanish": {
            "name": "西班牙文",
            "instruction": "请使用专业西班牙语输出，确保语法和表达符合西班牙语规范",
            "examples": ["inteligencia artificial", "aprendizaje automático", "modelo de lenguaje", "aprendizaje profundo"]
        },
        "russian": {
            "name": "俄文",
            "instruction": "请使用专业俄语输出，确保语法和表达符合俄语规范",
            "examples": ["искусственный интеллект", "машинное обучение", "языковая модель", "глубокое обучение"]
        }
    }
    
    # 如果目标语言未定义，默认使用中文
    lang_info = language_mapping.get(target_language, language_mapping["chinese"])
    
    # 构建思维链提示词
    cot_prompt = f"""请严格按照思维链(Chain of Thought)方法生成{lang_info['name']}{content_type}。
    
思维步骤：
1. 理解内容：首先深入理解原始内容的主题、关键点和上下文
2. 提取要点：识别最重要的信息、概念和术语
3. 语言转换：确保完全使用{lang_info['name']}表达，包括所有术语、概念和表达方式
4. 检查一致性：确保整个{content_type}在语言上保持一致，不混入其他语言

语言要求：
{lang_info['instruction']}

例如，以下术语应该使用{lang_info['name']}表达：
{', '.join(lang_info['examples'])}

最终输出：
请确保100%使用{lang_info['name']}输出，不得混入其他语言的单词或表达。即使是专业术语也必须使用{lang_info['name']}对应的表达方式。
"""
    
    return cot_prompt
class BM25Okapi:
    """BM25 搜索算法实现，优化用于手语识别和神经网络相关领域"""
    
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        """初始化BM25 Okapi搜索模型
        
        Args:
            corpus: 文档集合，每个文档是分词后的列表
            k1: 控制词频缩放的参数，通常在1.2-2.0之间
            b: 控制文档长度归一化的参数，通常为0.75
            epsilon: 平滑因子
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # 领域特定术语权重
        self.domain_weights = {
            "手语": 2.5,
            "识别": 2.0,
            "残差网络": 3.0,
            "resnet": 3.0,
            "神经网络": 2.5,
            "深度学习": 2.0,
            "cnn": 2.0,
            "卷积神经网络": 2.5
        }
        
        # 初始化文档频率和IDF值
        self._compute_idf()
    
    def _compute_idf(self):
        """计算文档频率和IDF值"""
        self.doc_count = len(self.corpus)
        self.avg_doc_len = sum(len(doc) for doc in self.corpus) / self.doc_count
        
        # 文档频率统计
        self.doc_freqs = []
        self.term_freqs = {}
        
        # 统计每个文档中词项的频率
        for doc in self.corpus:
            doc_freq = {}
            for term in doc:
                if term not in doc_freq:
                    doc_freq[term] = 0
                doc_freq[term] += 1
            
            self.doc_freqs.append(doc_freq)
            
            # 更新全局词项频率
            for term in doc_freq:
                if term not in self.term_freqs:
                    self.term_freqs[term] = 0
                self.term_freqs[term] += 1
        
        # 计算IDF值
        self.idf = {}
        for term, freq in self.term_freqs.items():
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + self.epsilon)
    
    def get_scores(self, query):
        """计算查询与文档集合的BM25相似度得分
        
        Args:
            query: 查询词项列表
            
        Returns:
            文档得分列表，与corpus中的文档顺序对应
        """
        # 扩展查询，增加领域特定术语
        expanded_query = self._expand_query(query)
        
        # 计算文档得分
        scores = [0.0] * self.doc_count
        
        for term in expanded_query:
            # 如果词项不在语料库中，跳过
            if term not in self.idf:
                continue
                
            term_weight = self.domain_weights.get(term.lower(), 1.0)
            
            # 计算该词项对每个文档的贡献
            for doc_id, doc_freq in enumerate(self.doc_freqs):
                if term not in doc_freq:
                    continue
                
                # 获取词项在当前文档中的频率
                freq = doc_freq[term]
                
                # 文档长度归一化
                doc_len = sum(doc_freq.values())
                doc_len_ratio = doc_len / self.avg_doc_len
                
                # BM25公式
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len_ratio)
                
                # 应用领域权重
                scores[doc_id] += (numerator / denominator) * term_weight
        
        return scores
    
    def _expand_query(self, query):
        """扩展查询，增加领域特定术语的相关近义词
        
        Args:
            query: 原始查询词项列表
            
        Returns:
            扩展后的查询词项列表
        """
        expanded = list(query)
        
        # 检查查询中是否包含手语识别相关术语
        query_text = " ".join(query).lower()
        
        # 手语识别相关术语扩展
        if "手语" in query_text:
            expanded.append("手势语言")
            if "识别" in query_text:
                expanded.append("手语识别")
                expanded.append("sign language recognition")
        
        # 残差网络相关术语扩展
        if "残差" in query_text or "resnet" in query_text:
            expanded.append("残差网络")
            expanded.append("resnet")
            if "识别" in query_text:
                expanded.append("残差网络识别")
        
        # 神经网络相关术语扩展
        if "神经" in query_text and "网络" in query_text:
            expanded.append("神经网络")
            expanded.append("neural network")
            expanded.append("深度神经网络")
            if "卷积" in query_text:
                expanded.append("卷积神经网络")
                expanded.append("cnn")
        
        return expanded

def preprocess_search_query(query):
    """预处理搜索查询，增强特定领域术语，提高对手语识别和神经网络等特定领域关键词的检索效果
    
    Args:
        query: 原始查询字符串
    
    Returns:
        增强后的查询字符串
    """
    if not query:
        return query
    
    # 查询小写化用于匹配检查
    query_lower = query.lower()
    
    # 手语识别领域特定术语
    if "手语" in query_lower and "识别" in query_lower:
        return f"{query} 手语识别 sign language recognition"
    
    if "手语" in query_lower:
        return f"{query} 手语 sign language"
        
    if "神经网络" in query_lower and "手语" in query_lower:
        return f"{query} 神经网络手语识别"
        
    if "残差网络" in query_lower or "resnet" in query_lower.lower():
        return f"{query} 残差网络 ResNet"
        
    # 返回原始查询
    return query

@app.route('/preview/<int:summary_id>', methods=['GET'])
@login_required
def preview_document(summary_id):
    """预览原文内容"""
    try:
        # 获取摘要记录
        summary = DocumentSummary.query.get(summary_id)
        if not summary:
            return render_template('404.html', message=f"未找到ID为 {summary_id} 的摘要记录"), 404
            
        # 检查用户权限 - 只有管理员和记录所有者可以预览
        current_user_id = session.get('user_id')
        current_user = User.query.get(current_user_id)
        if current_user.id != summary.user_id and current_user.role != 'admin':
            return render_template('error.html', message="您无权查看此文档"), 403
        
        # 获取文件类型
        file_ext = summary.file_name.split('.')[-1].lower() if '.' in summary.file_name else 'txt'
        file_type = 'txt'  # 默认类型
        
        # 判断文件类型
        if file_ext in ['pdf']:
            file_type = 'pdf'
        elif file_ext in ['docx', 'doc']:
            file_type = 'docx'
        elif file_ext in ['md', 'markdown']:
            file_type = 'md'
        elif file_ext in ['epub']:
            file_type = 'epub'
        
        # 处理分页请求
        page = request.args.get('page', 1, type=int)
        
        # 获取文件内容
        file_content = get_file_content(summary_id)
        
        # 确保内容不为空
        if not file_content:
            return render_template('error.html', message="无法获取文件内容"), 404
        
        # 处理二进制内容 - 尝试解码为文本
        if isinstance(file_content, bytes):
            try:
                # 尝试使用UTF-8解码
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他常见编码
                encodings = ['latin-1', 'gbk', 'gb2312', 'iso-8859-1']
                content = None
                for encoding in encodings:
                    try:
                        content = file_content.decode(encoding)
                        print(f"成功使用 {encoding} 解码文件内容")
                        break
                    except UnicodeDecodeError:
                        continue
                
                # 如果所有编码都失败，使用latin-1（不会引发解码错误）
                if not content:
                    content = file_content.decode('latin-1')
                    print("使用 latin-1 兜底解码文件内容")
        else:
            # 已经是文本格式
            content = file_content
            
        # 特殊文件类型处理
        if file_type == 'md' and not content.startswith('#'):
            # 检查是否为二进制形式的markdown内容（以b"开头）
            if content.startswith('b"') or content.startswith("b'"):
                try:
                    # 尝试解析Python字符串表示
                    content = content[2:-1]  # 移除b"和最后的"
                    # 处理转义序列
                    content = content.encode('latin-1').decode('unicode_escape')
                    print("成功解析二进制字符串表示的markdown内容")
                except Exception as e:
                    print(f"解析二进制字符串表示时出错: {str(e)}")
        
        # AJAX请求返回JSON格式的页面内容
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # 简单的分页处理，每页5000个字符
            page_size = 5000
            content_parts = [content[i:i+page_size] for i in range(0, len(content), page_size)]
            total_pages = len(content_parts)
            
            if page > total_pages:
                return jsonify({'error': '页面不存在'}), 404
            
            current_part = content_parts[page-1] if page <= total_pages else ''
            
            return jsonify({
                'content': current_part,
                'current_page': page,
                'total_pages': total_pages,
                'has_more': page < total_pages,
                'content_length': len(content)
            })
        
        # 非AJAX请求返回完整页面
        # 简单的分页处理，每页5000个字符
        page_size = 5000
        initial_content = content[:page_size] if content else ''
        total_pages = (len(content) + page_size - 1) // page_size if content else 1
        
        return render_template(
            'preview.html',
            summary=summary,
            filename=summary.file_name,
            file_type=file_type,
            initial_content=initial_content,
            total_pages=total_pages
        )
    except Exception as e:
        print(f"预览文件错误: {str(e)}")
        traceback.print_exc()
        return render_template('error.html', message=f"预览文件时出错: {str(e)}"), 500

@app.route('/api/check_vectorization/<int:doc_id>', methods=['GET'])
@login_required
def check_vectorization_status(doc_id):
    """检查文档向量化状态"""
    try:
        # 获取当前用户ID
        user_id = session.get('user_id')
        
        # 查询文档是否存在并且属于当前用户
        doc = DocumentSummary.query.filter_by(id=doc_id, user_id=user_id).first()
        if not doc:
            return jsonify({
                'success': False,
                'error': '文档不存在或无权访问'
            }), 404
            
        # 检查向量化状态
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'has_vector_store': doc.has_vector_store,
            'vectorization_complete': doc.has_vector_store,
            'vectorization_pending': not doc.has_vector_store,
            'chroma_collection': doc.chroma_collection
        })
        
    except Exception as e:
        logger.error(f"检查文档 {doc_id} 向量化状态时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'检查向量化状态失败: {str(e)}'
        }), 500

def process_vectorization_queue():
    """后台线程函数，处理向量化任务队列"""
    logger.info("向量化任务处理线程启动")
    while True:
        try:
            # 从队列获取任务
            task = vectorization_queue.get()
            if task is None:  # 检查是否是终止信号
                break
                
            doc_id = task.get('doc_id')
            text = task.get('text')
            summary = task.get('summary')
            
            logger.info(f"开始处理文档ID {doc_id} 的向量化任务")
            
            try:
                # 执行向量化处理
                success = create_hybrid_vector_store(text, summary, doc_id)
                logger.info(f"文档ID {doc_id} 的向量化处理{'成功' if success else '失败'}")
                
                # 更新数据库中的向量化状态
                if success:
                    try:
                        doc = DocumentSummary.query.get(doc_id)
                        if doc:
                            doc.has_vector_store = True
                            doc.chroma_collection = f"doc_{doc_id}"
                            db.session.commit()
                            logger.info(f"已更新文档ID {doc_id} 的向量存储状态")
                    except Exception as e:
                        logger.error(f"更新文档ID {doc_id} 向量存储状态失败: {str(e)}")
            except Exception as e:
                logger.error(f"处理文档ID {doc_id} 的向量化任务时出错: {str(e)}")
                traceback.print_exc()
            finally:
                # 标记任务完成
                vectorization_queue.task_done()
        except Exception as e:
            logger.error(f"向量化任务处理线程出错: {str(e)}")
            traceback.print_exc()

# 启动向量化处理线程
vectorization_thread = Thread(target=process_vectorization_queue, daemon=True)
vectorization_thread.start()

def async_create_vector_store(text, summary, doc_id):
    """异步创建向量存储，将任务添加到队列"""
    logger.info(f"添加文档ID {doc_id} 到向量化队列")
    vectorization_queue.put({
        'doc_id': doc_id,
        'text': text,
        'summary': summary
    })

if __name__ == '__main__':
    with app.app_context():
        # 只在表不存在时创建表
        db.create_all()
        # 初始化管理员账户和RAGTools
        init_admin()
        init_rag_tools()
        print("数据库和应用组件初始化完成")
    # 修改端口为5001或8080等非常用端口，并明确指定主机地址
    app.run(debug=True, host='127.0.0.1', port=8080)


