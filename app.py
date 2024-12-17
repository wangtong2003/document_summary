from flask import Flask, request, jsonify, Response, render_template, stream_with_context
import asyncio
from ollama import Client
import os
import PyPDF2
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
import jieba

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md', 'epub'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'

# MySQL配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:20030221@localhost/doc_summary?charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 创建上传目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 数据模型定义
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class DocumentSummary(db.Model):
    __tablename__ = 'document_summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(32), nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    summary_length = db.Column(db.String(20))
    target_language = db.Column(db.String(20))

def save_summary_to_db(file_info, summary_text, params):
    """
    保存或更新文档摘要到MySQL
    """
    try:
        print("开始保存摘要到数据库...")
        print(f"文件信息: {file_info}")
        print(f"摘要长度: {len(summary_text) if summary_text else 0}")
        print(f"参数: {params}")
        
        # 计算文件内容的MD5哈希值
        import hashlib
        file_hash = hashlib.md5(file_info["original_text"].encode('utf-8')).hexdigest()
        
        # 检查是否已存在相同文件的摘要
        existing_summary = DocumentSummary.query.filter_by(
            file_hash=file_hash
        ).first()
        
        if existing_summary:
            print(f"更新现有摘要 ID: {existing_summary.id}")
            # 更新现有摘要
            existing_summary.summary_text = summary_text
            existing_summary.summary_length = params.get("summary_length")
            existing_summary.target_language = params.get("target_language") 
            existing_summary.updated_at = datetime.now()
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
                target_language=params.get("target_language")
            )
            db.session.add(new_summary)
            db.session.commit()
            print("新摘要保存成功")
            
        return True
        
    except Exception as e:
        print(f"保存摘要到数据库时出错: {str(e)}")
        db.session.rollback()
        raise e

# 文档处理函数
def read_pdf(file_path):
    """读取PDF文件内容"""
    try:
        print(f"开始读取PDF文件: {file_path}")
        text = ""
        with open(file_path, 'rb') as file:
            # 创建PDF文件阅读器对象
            pdf_reader = PyPDF2.PdfReader(file)
            
            # 获取PDF文件页数
            num_pages = len(pdf_reader.pages)
            print(f"PDF文件共有 {num_pages} 页")
            
            # 遍历每一页并提取文本
            for page_num in range(num_pages):
                try:
                    # 获取页面对象
                    page = pdf_reader.pages[page_num]
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    print(f"成功读取第 {page_num + 1} 页，提取到 {len(page_text)} 个字符")
                except Exception as e:
                    print(f"读取第 {page_num + 1} 页时出错: {str(e)}")
                    continue
            
            if not text.strip():
                raise Exception("未能从PDF中提取到任何文本")
                
            print(f"PDF文件读取完成，总共提取到 {len(text)} 个字符")
            return text
    except Exception as e:
        print(f"读取PDF文件出错: {str(e)}")
        raise Exception(f"读取PDF文件失败: {str(e)}")

def read_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def read_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return markdown.markdown(text)

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

        [INSTRUCTIONS]
        - Generate a summary exactly matching the specified length;
        - Write in chinese;
        """
        
        content = f"""
        ARTICLE_TO_SUMMARY:
        <ARTICLE>
        {text}
        </ARTICLE>
        """
        
        print("调用远程 Ollama API...")
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

def ollama_text(input_text, params=None, file_info=None):
    try:
        client = Client(host='http://localhost:11434')
        
        # 根据参数调整提示词
        summary_length_map = {
            'very_short': '约100字',
            'short': '约200字',
            'medium': '约500字',
            'long': '约1000字',
            'very_long': '约2000字'
        }
        
        # 获取必需参数
        summary_length = params.get('summary_length', 'medium')
        target_length = summary_length_map.get(summary_length, '约500字')
        target_language = params.get('target_language', 'chinese')
        
        # 构建基础提示模板
        system_prompt = f"""你是一个专业的文档摘要助手。请仔细阅读以下内容，并生成结构化摘要。

        [输出格式要求]
        1. 使用标准Markdown格式
        2. 每个段落之间要有空行
        3. 代码块要使用独立的```标记并注明语言
        4. 列表项要使用统一的缩进
        5. 重要内容使用加粗标记
        6. 专业术语使用行内代码标记

        [文档结构]

        # 文档摘要分析

        ## 主要任务

        {input_text}

        ## 详细分析

        ### 1. 问题描述
        - **问题背景**：xxx
        - **问题现状**：xxx
        - **解决目标**：xxx

        ### 2. 解决方案
        1. **方案一**：xxx
           - 具体步骤
           - 实现方法
           - 关键代码

        2. **方案二**：xxx
           - 具体步骤
           - 实现方法
           - 关键代码

        ### 3. 代码实现
        ```python
        # 示例代码
        def example():
            # 实现逻辑
            pass
        ```

        ### 4. 注意事项
        - 重要提示1
        - 重要提示2
        - 重要提示3

        ## 总结要点
        1. xxx
        2. xxx
        3. xxx

        [注意事项]
        1. 保持专业性和准确性
        2. 突出文档的主要观点
        3. 保持格式的一致性
        4. 确保代码示例的完整性
        5. 适当使用加粗和引用格式

        [输出要求]
        - 语言：{target_language}
        - 长度：{target_length}
        - 专业程度：{params.get('expertise_level', 'intermediate')}
        - 风格：{params.get('language_style', 'neutral')}
        """
        
        content = f"""
        需要总结的文档内容：
        <文档>
        {input_text}
        </文档>
        """
        
        print("调用远程 Ollama API...")
        response = client.generate(
            model='qwen2.5:3b',
            prompt=system_prompt + content,
            stream=True
        )
        
        def generate():
            complete_summary = []
            for chunk in response:
                if chunk and 'response' in chunk:
                    chunk_text = chunk['response']
                    complete_summary.append(chunk_text)
                    yield f"data: {chunk_text}\n\n"
            
            # 在流式输出完成后，保存完整的摘要到数据库
            if file_info:
                with app.app_context():
                    try:
                        full_summary = ''.join(complete_summary)
                        save_summary_to_db(file_info, full_summary, params)
                        print("摘要保存成功")
                    except Exception as e:
                        print(f"保存摘要失败: {str(e)}")
                    
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        print(f"Ollama API错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

def init_admin():
    """初始化管理员账户"""
    try:
        print("检查管理员账户...")
        # 检查管理员是否已存在
        admin = User.query.filter_by(username='admin').first()
        
        if not admin:
            print("创建新的管理员账户...")
            # 创建管理员账户
            admin = User(
                id=str(uuid.uuid4()),
                username='admin',
                password=generate_password_hash('admin'),
                email='admin@example.com',
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("管理员账户创建成功")
        else:
            print("管理员账户已存在")
    except Exception as e:
        print(f"初始化管理员账户错误: {str(e)}")
        db.session.rollback()
        raise

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
    """渲染摘要库页面"""
    return render_template('summaries.html')

@app.route('/summaries', methods=['GET'])
def get_summaries():
    """获取所有摘要列表"""
    try:
        print("\n=== 开始获取摘要列表 ===")
        summaries = DocumentSummary.query.order_by(DocumentSummary.created_at.desc()).all()
        print(f"查询到 {len(summaries)} 条摘要记录")
        
        result = [{
            'id': s.id,
            'file_name': s.file_name,
            'file_type': s.file_name.split('.')[-1] if '.' in s.file_name else 'unknown',
            'summary_text': s.summary_text,
            'created_at': s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': s.summary_length,
            'target_language': s.target_language
        } for s in summaries]
        
        print(f"返回 {len(result)} 条摘要记录")
        return jsonify(result)
        
    except Exception as e:
        print(f"获取摘要列表错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summaries/<int:summary_id>', methods=['GET'])
def get_summary_detail(summary_id):
    """获取单个摘要详情"""
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
        summary = DocumentSummary.query.get_or_404(summary_id)
        db.session.delete(summary)
        db.session.commit()
        return jsonify({'message': '删除成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

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

@app.route('/process_document', methods=['POST'])
def process_document():
    """处理上传的文档并生成摘要"""
    try:
        print("\n=== 开始处理文档 ===")
        
        # 检查是否有文件被上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
            
        # 获取参数
        params = {
            'summary_length': request.form.get('summary_length', 'medium'),
            'target_language': request.form.get('target_language', 'chinese'),
            'summary_style': request.form.get('summary_style'),
            'focus_area': request.form.get('focus_area'),
            'expertise_level': request.form.get('expertise_level'),
            'language_style': request.form.get('language_style')
        }
        
        print(f"原始文件名: {file.filename}")
        
        # 使用自定义的sanitize_filename函数处理文件名
        safe_filename = sanitize_filename(file.filename)
        print(f"处理后的安全文件名: {safe_filename}")
        
        # 检查文件类型
        if not allowed_file(safe_filename):
            print(f"文件扩展名检查失败: {safe_filename}")
            return jsonify({'error': '不支持的文件格式'}), 400
            
        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        print(f"文件保存路径: {file_path}")
        file.save(file_path)
        
        try:
            # 读取文档内容
            text = read_document(file_path)
            
            # 准备文件信息
            file_info = {
                'filename': safe_filename,
                'original_text': text
            }
            
            # 生成摘要
            return ollama_text(text, params, file_info)
            
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        print(f"处理文档时发生错误: {str(e)}")
        return jsonify({'error': f'处理文档失败: {str(e)}'}), 500

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    extension = os.path.splitext(filename)[1].lower()
    print(f"检查文件扩展名: {extension}")
    print(f"允许的扩展名: {ALLOWED_EXTENSIONS}")
    return extension[1:] in ALLOWED_EXTENSIONS

class ChineseTokenizer:
    def __call__(self, text):
        return list(jieba.cut(text))

def search_summaries(query, summaries, top_k=5):
    """
    使用TF-IDF和余弦相似度进行智能检索
    
    Args:
        query: 搜索查询
        summaries: 摘要列表
        top_k: 返回的最相关结果数量
        
    Returns:
        最相关的文档列表，包含相似度分数
    """
    if not summaries:
        return []
        
    # 准备文档集合
    documents = [summary.summary_text for summary in summaries]
    documents.append(query)  # 将查询添加到文档集合中
    
    # 创建TF-IDF向量化器，使用自定义的中文分词器
    vectorizer = TfidfVectorizer(
        tokenizer=ChineseTokenizer(),
        stop_words='english',  # 移除英文停用词
        max_features=5000  # 限制特征数量
    )
    
    try:
        # 转换文档为TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # 计算查询（最后一个文档）与其他文档的余弦相似度
        cosine_similarities = cosine_similarity(
            tfidf_matrix[-1:], tfidf_matrix[:-1]
        ).flatten()
        
        # 获取相似度最高的文档索引
        top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        
        # 构建结果列表
        results = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:  # 只返回相似度大于0的结果
                summary = summaries[idx]
                results.append({
                    'id': summary.id,
                    'file_name': summary.file_name,
                    'summary_text': summary.summary_text,
                    'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'similarity_score': float(cosine_similarities[idx]),
                    'summary_length': summary.summary_length,
                    'target_language': summary.target_language
                })
        
        return results
    except Exception as e:
        print(f"搜索过程中出错: {str(e)}")
        return []

@app.route('/api/search', methods=['POST'])
def search_api():
    """智能检索API端点"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': '搜索查询不能为空'}), 400
            
        # 获取所有摘要
        summaries = DocumentSummary.query.all()
        
        # 执行智能检索
        results = search_summaries(query, summaries)
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        print(f"搜索API错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("正在启动应用程序...")
    try:
        with app.app_context():
            print("正在初始化数据库...")
            # 创建所有数据库表
            db.create_all()
            print("数据库表创建成功")
            # 初始化管理员账户
            init_admin()
            print("管理员账户初始化成功")
        print("正在启动 Flask 服务器...")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"启动应用程序时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
