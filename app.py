from flask import Flask, request, jsonify, Response, render_template, stream_with_context, session, redirect, url_for, make_response
from werkzeug.security import generate_password_hash, check_password_hash
import asyncio
from ollama import AsyncClient, Client
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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'epub'}

# JWT配置
app.config['JWT_SECRET_KEY'] = '3e8f44d5b59f969e020f82c088bf0c56bdccc1a13db66ac013edbb24dce572a7'  # 建议使用复杂的密钥
app.config['JWT_TOKEN_LOCATION'] = ['headers']     # 指定从请求头中获取token
app.config['JWT_HEADER_NAME'] = 'Authorization'    # 指定请求头的名称
app.config['JWT_HEADER_TYPE'] = 'Bearer'          # 指定token前缀
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
# 在 Flask 应用配置中添加
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'

jwt = JWTManager(app)

# 添加JWT错误处理
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        'status': 401,
        'sub_status': 42,
        'msg': 'Token已过期'
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        'status': 401,
        'sub_status': 43,
        'msg': '无效的Token'
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({
        'status': 401,
        'sub_status': 44,
        'msg': '缺少Token'
    }), 401

# 添加配置
app.config['SECRET_KEY'] = '3e8f44d5b59f969e020f82c088bf0c56bdccc1a13db66ac013edbb24dce572a7'  # 用于 JWT
app.config['JWT_EXPIRATION_DELTA'] = timedelta(days=1)

# 数据库配置
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'doc_summary'
}

# 创建上传目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 数据库连接函数
def get_db_connection():
    return mysql.connector.connect(**db_config)


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

def ollama_text(input_text, model='qwen2.5:3b'):
    try:
        client = Client(host='http://localhost:11434')
        # 构建提示模板
        system_prompt = """You are a powerful AI assistant capable of reading text content and summarizing it. The purpose of the summary is to help users grasp the key points without reading the entire article.

        [INSTRUCTIONS]
        - Write a concise summary within 800 characters, capturing the essence of the article in one sentence;
        - Outline the article in up to 10 points, with each point being less than 120 characters;
        - Utilize only factual information from the original text. Refrain from fabricating content;
        - Please use Arabic numerals for point numbering;
        - Before finalizing, check the word count to ensure it is within the length limitation;
        - If the text exceeds the length limitation, trim redundant parts;
        - Do not use markdown format;
        
        <OUTPUT_FORMAT_1>
        This is an example summary.
        1.Point one.[P2]
        2.Point two.[P2, P3]
        ...
        </OUTPUT_FORMAT_1>

        <OUTPUT_FORMAT_2>
        这是一个示例总结。
        1.要点一。[P1]
        2.要点二。[P1, P2, P3]
        ...
        </OUTPUT_FORMAT_2>
        """
        content=f"""
        ARTICLE_TO_SUMMARY:
        <ARTICLE>
        {input_text}
        </ARTICLE>


        **Note:**
        1.Please repeatedly check and confirm that the page numbers in the output match the page numbers in the input. 
        2.Note that the summary should not be marked with page numbers, but each core event needs to be marked with page numbers.
        3.Please re-check, if you do not meet the above three points, please revise it immediately
        """
        def generate():
            response = client.generate(
                model=model,
                prompt=system_prompt,
                stream=True
            )
            
            for chunk in response:
                if chunk and 'response' in chunk:
                    # 使用 SSE 格式返回数据
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
    # 更严格的文件名检查
    if not filename or '.' not in filename:
        return False
    # 获取最后一个点后面的扩展名
    extension = filename.rsplit('.', 1)[-1].lower()
    return extension in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '未登录'}), 401
            
        try:
            payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = payload
            return f(*args, **kwargs)
        except jwt_exceptions.ExpiredSignatureError:
            return jsonify({'error': 'token已过期'}), 401
        except jwt_exceptions.InvalidTokenError:
            return jsonify({'error': '无效的token'}), 401
            
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            current_user = get_jwt_identity()
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
            user = cursor.fetchone()
            
            if not user or user['role'] != 'admin':
                return jsonify({'error': '需要管理员权限'}), 403
                
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': str(e)}), 401
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    return decorated_function

@app.errorhandler(404)
def not_found_error(error):
    print(f"404错误: {error}")  # 添加日志
    return jsonify({'error': '请求的页面不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"500错误: {error}")  # 添加日志
    return jsonify({'error': '服务器内部错误'}), 500
@app.after_request
def after_request(response):
    # 对所有 JSON 响应设置正确的编码
    if response.mimetype == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        
        # 打印响应内容用于调试
        print('Response headers:', dict(response.headers))
        print('Response data:', response.get_data(as_text=True))
        
    return response

@app.errorhandler(Exception)
def handle_error(error):
    print(f"发生错误: {str(error)}")
    response = jsonify({
        'error': str(error),
        'success': False
    })
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response, 500

# 路由
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

def get_secure_filename(filename):
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)
    if not ext:
        return None, None
        
    # 处理文件名，保留扩展名
    secure_name = secure_filename(name)
    # 如果文件名被完全过滤掉了，使用时间戳作为文件名
    if not secure_name:
        secure_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 返回安全的文件名和原始扩展名
    return secure_name, ext.lower()[1:]  # 移除扩展名中的点

@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    try:
        if 'file' not in request.files:
            print("错误：请求中没有文件")
            return jsonify({'error': '没有文件被上传'}), 400
            
        file = request.files['file']
        if not file or not file.filename:
            print("错误：文件名为空")
            return jsonify({'error': '没有选择文件'}), 400

        # 打印原始文件名用于调试
        print(f"原始文件名: {file.filename}")
        
        # 处理文件名
        secure_name, file_extension = get_secure_filename(file.filename)
        if not secure_name or not file_extension:
            print("错误：无效的文件名或扩展名")
            return jsonify({'error': '无效的文件名或扩展名'}), 400
            
        print(f"处理后的文件名: {secure_name}")
        print(f"文件扩展名: {file_extension}")
        
        # 验证文件类型
        if file_extension not in ALLOWED_EXTENSIONS:
            print(f"错误：不支持的文件格式 {file_extension}")
            return jsonify({'error': f'不支持的文件格式: {file_extension}'}), 400

        # 获取当前用户ID
        current_user_id = get_jwt_identity()
            
        # 生成完整的文件名
        full_filename = f"{secure_name}_{str(uuid.uuid4())[:8]}.{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, full_filename)
        
        # 保存文件
        file.save(file_path)
        print(f"文件已保存到: {file_path}")

        try:
            # 读取文件内容
            text = ''
            if file_extension == 'pdf':
                text = read_pdf(file_path)
            elif file_extension == 'docx':
                text = read_docx(file_path)
            elif file_extension == 'txt':
                text = read_txt(file_path)
            elif file_extension == 'md':
                text = read_md(file_path)
            elif file_extension == 'epub':
                text = read_epub(file_path)
            else:
                raise ValueError(f'不支持的文件格式: {file_extension}')

            if not text.strip():
                raise ValueError('文件内容为空')

            # 调用 AI 处理并获取响应
            client = Client(host='http://localhost:11434')
            response = client.chat(model='qwen2.5:3b', messages=[
                {
                    'role': 'user',
                    'content': f'请为以下文本生成一个详细的摘要：\n\n{text}'
                }
            ])

            if not response or 'message' not in response:
                raise ValueError('AI 处理失败')

            # 确保内容是 UTF-8 编码的字符串
            content = response['message']['content']
            
            # 使用 make_response 和 jsonify 来设置正确的响应头
            resp = make_response(jsonify({
                'success': True,
                'content': content
            }))
            
            # 设置响应头
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            return resp

        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"临时文件已删除: {file_path}")

    except Exception as e:
        print(f"处理文件错误: {str(e)}")
        error_resp = make_response(jsonify({
            'error': str(e),
            'success': False
        }))
        error_resp.headers['Content-Type'] = 'application/json; charset=utf-8'
        return error_resp, 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


# 错误处理
@app.errorhandler(Exception)
def handle_error(error):
    print(f"发生错误: {str(error)}")
    return jsonify({'error': str(error)}), 500

# 用户注册
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    
    # 不允许注册管理员账户
    if data.get('username') == 'admin':
        return jsonify({'error': '不允许使用此用户名'}), 400
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = %s", (data['username'],))
        if cursor.fetchone():
            return jsonify({'error': '用户名已存在'}), 400
            
        # 检查邮箱是否已存在
        cursor.execute("SELECT id FROM users WHERE email = %s", (data['email'],))
        if cursor.fetchone():
            return jsonify({'error': '邮箱已被注册'}), 400
            
        # 创建新用户，强制设置为普通用户
        user_id = str(uuid.uuid4())
        hashed_password = generate_password_hash(data['password'])
        cursor.execute(
            """INSERT INTO users (id, username, password, email, role)
               VALUES (%s, %s, %s, %s, 'user')""",
            (user_id, data['username'], hashed_password, data['email'])
        )
        conn.commit()
        
        return jsonify({
            'message': '注册成功',
            'user': {
                'id': user_id,
                'username': data['username'],
                'email': data['email'],
                'role': 'user'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 用户认证路由
@app.route('/api/user')
@jwt_required()
def get_user():
    try:
        current_user_id = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT id, username, role FROM users WHERE id = %s", (current_user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify({
            'id': user['id'],
            'username': user['username'],
            'role': user['role']
        })
        
    except Exception as e:
        print(f"获取用户信息错误: {str(e)}")
        return jsonify({'error': '获取用户信息失败'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 登录路由
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            access_token = create_access_token(identity=user['id'])
            return jsonify({
                'token': access_token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role']
                }
            })
        else:
            return jsonify({'error': '用户名或密码错误'}), 401
            
    except Exception as e:
        print(f"登录错误: {str(e)}")
        return jsonify({'error': '登录失败'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 用户注销
@app.route('/api/logout', methods=['POST'])
def logout():
    # 前端需要清除 token
    return jsonify({'message': '注销成功'})

# 获取用户信
@app.route('/api/user/profile')
def get_user_profile():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': '未登录'}), 401
        
    try:
        # 验证 token
        payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT id, username, email, role FROM users WHERE id = %s",
            (user_id,)
        )
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': '用不存在'}), 404
            
        return jsonify(user)
    except jwt_exceptions.ExpiredSignatureError:
        return jsonify({'error': 'token已过期'}), 401
    except jwt_exceptions.InvalidTokenError:
        return jsonify({'error': '无效的token'}), 401
    except Exception as e:
        print(f"获取用户信息错误: {str(e)}")  # 添加调试日志
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 收藏消息
@app.route('/api/favorites/add', methods=['POST'])
def add_favorite():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': '未登录'}), 401
        
    try:
        payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
        
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        favorite_id = str(uuid.uuid4())
        cursor.execute(
            """INSERT INTO favorites (id, user_id, message_id)
               VALUES (%s, %s, %s)""",
            (favorite_id, user_id, data['message_id'])
        )
        conn.commit()
        
        return jsonify({'message': '收藏成'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 获取收藏列表
@app.route('/api/favorites')
def get_favorites():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': '未登录'}), 401
        
    try:
        payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT f.*, m.content, m.created_at as message_date
               FROM favorites f
               JOIN messages m ON f.message_id = m.id
               WHERE f.user_id = %s
               ORDER BY f.created_at DESC""",
            (user_id,)
        )
        favorites = cursor.fetchall()
        
        return jsonify(favorites)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# 获取当前用户信息
@app.route('/api/user')
@login_required
def get_current_user():
    token = request.headers.get('Authorization')
    try:
        token = token.split(' ')[1]
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT id, username, email, role, created_at 
               FROM users 
               WHERE id = %s""",
            (payload['user_id'],)
        )
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify(user)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 用户管理页面路由
@app.route('/admin/users')
def admin_users():
    return render_template('admin/users.html')

# 获取用户列表 API
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, role, created_at FROM users")
        users = cursor.fetchall()
        return jsonify(users)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 更新用户角色 API
@app.route('/api/users/<user_id>/role', methods=['PUT'])
def update_user_role(user_id):
    try:
        data = request.get_json()
        new_role = data.get('role')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET role = %s WHERE id = %s",
            (new_role, user_id)
        )
        conn.commit()
        return jsonify({'message': '角色更新成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回空响应，状态码204表示无容

def simple_auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({
                'status': 401,
                'sub_status': 44,
                'msg': '缺少Token'
            }), 401
            
        return f(*args, **kwargs)
    return decorated_function

# 添加新的数据模型相关代码
class DocumentStatus:
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'

# 修改摘要库页面路由，移除强制认证
@app.route('/summaries')
def summaries_page():
    return render_template('summaries.html')

# 修改保存摘要的路由
@app.route('/api/save-summary', methods=['POST'])
@jwt_required()
def save_summary():
    try:
        current_user = get_jwt_identity()
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 添加更多字段以支持文档管理功能
        cursor.execute('''
            INSERT INTO summaries (
                user_id, 
                file_name, 
                file_type,
                content, 
                status,
                is_opened,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ''', (
            current_user,
            data['file_name'],
            data['file_type'],
            data['content'],
            DocumentStatus.COMPLETED,
            False
        ))
        
        summary_id = cursor.lastrowid
        conn.commit()
        
        return jsonify({
            'message': '保存成功',
            'summary_id': summary_id
        })
        
    except Exception as e:
        print(f"保存摘要错误: {str(e)}")
        return jsonify({'error': '保存失败'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 修改获取摘要列表的API，优化认证处理
@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify([])  # 如果没有token，返回空列表而不是错误
            
        # 解析token获取用户ID
        try:
            payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = payload['sub']  # JWT标准使用'sub'存用户ID
        except:
            return jsonify([])  # token无效时返回空列表
            
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute('''
            SELECT 
                id,
                file_name,
                file_type,
                content,
                status,
                is_opened,
                created_at
            FROM summaries 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        ''', (current_user,))
        
        summaries = cursor.fetchall()
        
        # 格式化日期时间
        for summary in summaries:
            if summary['created_at']:
                summary['created_at'] = summary['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(summaries)
        
    except Exception as e:
        print(f"获取摘要列表错误: {str(e)}")
        return jsonify([])  # 发生错误时返回空列表
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 获取单个摘要详情
@app.route('/api/summaries/<int:summary_id>', methods=['GET'])
@jwt_required()
def get_summary(summary_id):
    try:
        current_user = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute('''
            SELECT * FROM summaries 
            WHERE id = %s AND user_id = %s
        ''', (summary_id, current_user))
        
        summary = cursor.fetchone()
        
        if not summary:
            return jsonify({'error': '摘要不存在'}), 404
            
        # 更新打开状态
        cursor.execute('''
            UPDATE summaries 
            SET is_opened = TRUE 
            WHERE id = %s
        ''', (summary_id,))
        
        conn.commit()
        
        if summary['created_at']:
            summary['created_at'] = summary['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"获取摘要详情错误: {str(e)}")
        return jsonify({'error': '获取摘要详情失败'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 删除摘要
@app.route('/api/summaries/<int:summary_id>', methods=['DELETE'])
@jwt_required()
def delete_summary(summary_id):
    try:
        current_user = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM summaries 
            WHERE id = %s AND user_id = %s
        ''', (summary_id, current_user))
        
        conn.commit()
        
        if cursor.rowcount == 0:
            return jsonify({'error': '摘要不存在或无权限删除'}), 404
            
        return jsonify({'message': '删除成功'})
        
    except Exception as e:
        print(f"删除摘要错误: {str(e)}")
        return jsonify({'error': '删除摘要失败'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 初始化管理员账户的函数
def init_admin():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 检查管理员是否已存在
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        admin = cursor.fetchone()
        
        if not admin:
            # 创建管理员账户
            admin_id = str(uuid.uuid4())
            hashed_password = generate_password_hash('admin')
            cursor.execute(
                """INSERT INTO users (id, username, password, email, role)
                   VALUES (%s, %s, %s, %s, %s)""",
                (admin_id, 'admin', hashed_password, 'admin@example.com', 'admin')
            )
            conn.commit()
            print("管理员账户创建成功")
    except Exception as e:
        print(f"初始化管理员账户错误: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# 在应用动时初化管理员账户
if __name__ == '__main__':
    init_admin()  # 添加这行
    app.run(debug=True, port=5000)

