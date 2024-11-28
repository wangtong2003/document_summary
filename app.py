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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md', 'epub'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'

# MySQL配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/doc_summary?charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 创建上传目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 添加 User 模型定义
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

# 定义文档摘要模型
class DocumentSummary(db.Model):
    __tablename__ = 'document_summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), unique=True, nullable=False)  # 用于唯一标识文件
    summary_text = db.Column(db.Text, nullable=False)
    summary_length = db.Column(db.String(50))  # 摘要长度设置
    target_language = db.Column(db.String(50))  # 目标语言
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

def save_summary_to_db(file_info, summary_text, params):
    """
    保存或更新文档摘要到MySQL
    """
    try:
        # 检是否已存在相同文件的摘要
        existing_summary = DocumentSummary.query.filter_by(
            file_hash=file_info["file_hash"]
        ).first()
        
        if existing_summary:
            # 更新现有摘要
            existing_summary.summary_text = summary_text
            existing_summary.summary_length = params.get("summary_length")
            existing_summary.target_language = params.get("target_language") 
            existing_summary.updated_at = datetime.now()
            db.session.commit()
        else:
            # 创建新摘要记录
            new_summary = DocumentSummary(
                file_name=file_info["filename"],
                file_hash=file_info["file_hash"],
                summary_text=summary_text,
                summary_length=params.get("summary_length"),
                target_language=params.get("target_language")
            )
            db.session.add(new_summary)
            db.session.commit()
            
        return True
        
    except Exception as e:
        print(f"保存摘要错误: {str(e)}")
        db.session.rollback()
        return False

# 文档处理函数
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text

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
    elif file_extension == '.docx':
        return read_docx(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    elif file_extension == '.md':
        return read_md(file_path)
    elif file_extension == '.epub':
        return read_epub(file_path)
    else:
        raise ValueError("不支持的文件格式")

def ollama_text(input_text, params=None, file_info=None):
    try:
        client = Client(host='http://52bae7eb.r8.cpolar.top')
        
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
        system_prompt = f"""You are a powerful AI assistant capable of reading text content and summarizing it. 
        Please follow these basic requirements:

        [BASIC REQUIREMENTS]
        - Summary Length: {target_length}
        - Target Language: {target_language}

        [INSTRUCTIONS]
        - Generate a summary exactly matching the specified length ({target_length});
        - Write in {target_language};
        """
        
        # 根据可选参数添加额外指令
        if 'summary_style' in params:
            system_prompt += f"\n- Use {params['summary_style']} style;"
            
        if 'focus_area' in params:
            system_prompt += f"\n- Focus on {params['focus_area']} aspects;"
            
        if 'expertise_level' in params:
            system_prompt += f"\n- Match {params['expertise_level']} expertise level;"
            
        if 'language_style' in params:
            system_prompt += f"\n- Use {params['language_style']} tone;"
        
        # 添加输出格式
        system_prompt += """
        
        <OUTPUT_FORMAT>
        总体概述：
        [字数严格控制在{target_length}的整体总结]

        关键要点：
        1. [要点一]
        2. [要点二]
        ...
        </OUTPUT_FORMAT>
        """
        
        content = f"""
        ARTICLE_TO_SUMMARY:
        <ARTICLE>
        {input_text}
        </ARTICLE>
        """
        
        def generate():
            response = client.generate(
                model='qwen2.5:14b-instruct',
                prompt=system_prompt + content,
                stream=True
            )
            
            for chunk in response:
                if chunk and 'response' in chunk:
                    yield f"data: {chunk['response']}\n\n"
                    
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

def allowed_file(filename):
    if not filename or '.' not in filename:
        print(f"文件名无效或没有扩展名: {filename}")
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    is_allowed = extension in ALLOWED_EXTENSIONS
    if not is_allowed:
        print(f"不支持的文件扩展名: {extension}，支持的扩展名: {ALLOWED_EXTENSIONS}")
    return is_allowed

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/users', methods=['GET'])
def get_users_list():
    try:
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at
        } for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 用户注册
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    
    if data.get('username') == 'admin':
        return jsonify({'error': '不允许使用此用户名'}), 400
        
    try:
        # 检查用户名和邮箱
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': '用户名已存在'}), 400
            
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': '邮箱已被注册'}), 400
            
        # 创建新用户
        new_user = User(
            id=str(uuid.uuid4()),
            username=data['username'],
            password=generate_password_hash(data['password']),
            email=data['email'],
            role='user'
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'message': '注册成功',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email,
                'role': new_user.role
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# 用户认证路由
@app.route('/api/user')
def get_user():
    try:
        current_user_id = get_jwt_identity()
        user = db.session.get(User, current_user_id)
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify({
            'id': user.id,
            'username': user.username,
            'role': user.role,
            'email': user.email
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"获取用户信息错误: {str(e)}")
        return jsonify({'error': '获取用户信息失败'}), 500

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # 简单验证 - 任何用户名密码都可以登录
            return jsonify({
                'user_id': '1',
                'username': username,
                'role': 'admin'
            }), 200
                
        except Exception as e:
            print(f"登录错误: {str(e)}")
            return jsonify({'error': '登录失败'}), 500

# 用户注销
@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        # 这里可以添加任何需要的清理操作
        # 比如将token加入黑名单等
        return jsonify({
            'message': '注销成功',
            'success': True
        })
    except Exception as e:
        print(f"注销错误: {str(e)}")
        return jsonify({
            'error': '注销失败',
            'success': False
        }), 500

# 获取用户信息
@app.route('/api/user/profile')
def get_user_profile():
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        
        # 使用 SQLAlchemy 查询用户
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        })
    except Exception as e:
        print(f"获取用户信息错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 收藏消息
@app.route('/api/favorites/add', methods=['POST'])
def add_favorite():
    try:
        current_user_id = get_jwt_identity()
        data = request.json
        
        # 创建新的收藏
        new_favorite = Favorite(
            user_id=current_user_id,
            message_id=data['message_id']
        )
        
        db.session.add(new_favorite)
        db.session.commit()
        
        return jsonify({'message': '收藏成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# 获取收藏列表
@app.route('/api/favorites')
def get_favorites():
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        
        # 使用 SQLAlchemy 查询收藏
        favorites = db.session.query(
            Favorite, Message
        ).join(
            Message, Favorite.message_id == Message.id
        ).filter(
            Favorite.user_id == current_user_id
        ).order_by(
            Favorite.created_at.desc()
        ).all()
        
        return jsonify([{
            'id': fav.Favorite.id,
            'message_id': fav.Favorite.message_id,
            'content': fav.Message.content,
            'created_at': fav.Favorite.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'message_date': fav.Message.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for fav in favorites])
        
    except Exception as e:
        print(f"获取收藏列表错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# 获取当前用户信息
@app.route('/api/user')
def get_current_user():
    """获取当前用户信息"""
    return jsonify({
        'id': '1',
        'username': 'admin',
        'role': 'admin'
    })

# 用户管理页面路由
@app.route('/admin/users')
def admin_users():
    return render_template('admin/users.html')

# 获取用户列表 API
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at
        } for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 更新用户色 API
@app.route('/api/users/<user_id>/role', methods=['PUT'])
def update_user_role(user_id):
    try:
        data = request.get_json()
        new_role = data.get('role')
        
        user = User.query.get_or_404(user_id)
        user.role = new_role
        db.session.commit()
        
        return jsonify({'message': '角色更新成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回空响应，状态码204表示无容

# 添加新的数据模型相关代码
class DocumentStatus:
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'

# 改摘要库页面路由
@app.route('/summary_library')
def summary_library():
    """渲染摘要库页面"""
    return render_template('summaries.html')  # 使用 summaries.html 模板

# 获取摘要列表路由
@app.route('/summaries', methods=['GET'])
def get_summaries():
    """获取所有摘要史"""
    try:
        summaries = DocumentSummary.query.order_by(DocumentSummary.created_at.desc()).all()
        return jsonify([{
            'id': s.id,
            'file_name': s.file_name,
            'file_type': s.file_name.split('.')[-1] if '.' in s.file_name else 'unknown',
            'summary_text': s.summary_text,
            'created_at': s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': s.summary_length,
            'target_language': s.target_language
        } for s in summaries])
    except Exception as e:
        print(f"获取摘要列表错误: {str(e)}")
        return jsonify([])

# 获取单个摘要详情
@app.route('/summaries/<int:summary_id>', methods=['GET'])
def get_summary_detail(summary_id):
    """获取个摘要详情"""
    try:
        summary = DocumentSummary.query.get_or_404(summary_id)
        return jsonify({
            'id': summary.id,
            'file_name': summary.file_name,
            'summary_text': summary.summary_text,
            'created_at': summary.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_length': summary.summary_length,
            'target_language': summary.target_language
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# 删除摘要
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

# 初始化管理员账户的函数
def init_admin():
    """初始化管理员账户"""
    try:
        # 检查管理员是否已存在
        admin = User.query.filter_by(username='admin').first()
        
        if not admin:
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
            print("管理员账户创成功")
    except Exception as e:
        db.session.rollback()
        print(f"初始化管理员账户误: {str(e)}")

# 添加 Favorite 模型
class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    message_id = db.Column(db.Integer, db.ForeignKey('messages.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 添加 Message 模型
class Message(db.Model):
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 修改文件处理路由
@app.route('/process_document', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            print("没有文件被上传")
            return jsonify({'error': '没有文件被上传'}), 400
            
        file = request.files['file']
        if not file or not file.filename:
            print("没有选择文件")
            return jsonify({'error': '没有选择文件'}), 400
        
        print(f"接收到文件: {file.filename}")
            
        if not allowed_file(file.filename):
            print(f"不支持的文件格式: {file.filename}")
            return jsonify({'error': '不支持的文件格式'}), 400

        # 安全地处理文件名
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"保存文件到: {file_path}")
        file.save(file_path)

        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"成功读取文件内容，长度: {len(text)}")
            
            # 获取表单参数
            params = {
                'summary_length': request.form.get('summary_length', 'medium'),
                'target_language': request.form.get('target_language', 'chinese')
            }
            
            # 添加可选参数
            optional_params = ['summary_style', 'focus_area', 'expertise_level', 'language_style']
            for param in optional_params:
                if param in request.form:
                    params[param] = request.form[param]
            
            print(f"处理参数: {params}")  # 添加日志
            
            # 文件信息
            file_info = {
                'filename': filename,
                'file_path': file_path
            }

            # 调用 Ollama 处理文本
            return ollama_text(text, params, file_info)
            
        except Exception as e:
            print(f"读取文件内容错误: {str(e)}")
            raise
        finally:
            # 处理完后删除文件
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"删除临时文件: {file_path}")

    except Exception as e:
        print(f"处理文档错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 在应用启动时初始化管理员账户
if __name__ == '__main__':
    with app.app_context():
        # 创建所有数据库表
        db.create_all()
        # 初始化管理员账户
        init_admin()
    app.run(debug=True, port=5000)
