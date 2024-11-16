from flask import Flask, request, jsonify, Response, render_template, stream_with_context, session
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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'epub'}

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

def ollama_text(input_text, model='qwen:7b'):
    try:
        client = Client(host='http://localhost:11434')
        response = client.chat(model=model, messages=[
            {
                'role': 'user',
                'content': input_text
            }
        ])
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Ollama服务错误: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'token已过期'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': '无效的token'}), 401
            
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '未登录'}), 401
            
        try:
            payload = jwt.decode(token.split(' ')[1], app.config['SECRET_KEY'], algorithms=['HS256'])
            if payload['role'] != 'admin':
                return jsonify({'error': '需要管理员权限'}), 403
            request.user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'token已过期'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': '无效的token'}), 401
            
    return decorated_function

# 路由
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    session_id = request.form.get('sessionId')  # 获取会话ID
    
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        def generate():
            try:
                # 读取文档内容
                document_content = read_document(filename)
                
                # 保存用户的上传消息
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(
                        """INSERT INTO messages (session_id, content, is_user)
                           VALUES (%s, %s, %s)""",
                        (session_id, f"上传文件: {file.filename}", True)
                    )
                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()
                
                # 使用同步客户端
                client = Client(host='http://localhost:11434')
                response = client.chat(
                    model='qwen:7b',
                    messages=[{
                        'role': 'user',
                        'content': ollama_text(document_content)
                    }],
                    stream=True
                )
                
                accumulated_response = ""
                # 处理流式响应
                for part in response:
                    if part and 'message' in part and 'content' in part['message']:
                        content = part['message']['content']
                        accumulated_response += content
                        yield content
                
                # 保存AI的响应消息
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(
                        """INSERT INTO messages (session_id, content, is_user)
                           VALUES (%s, %s, %s)""",
                        (session_id, accumulated_response, False)
                    )
                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()
                
            except Exception as e:
                error_message = f"发生错误：{str(e)}"
                yield error_message
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(
                        """INSERT INTO messages (session_id, content, is_user)
                           VALUES (%s, %s, %s)""",
                        (session_id, error_message, False)
                    )
                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()
            finally:
                # 清理临时文件
                if os.path.exists(filename):
                    os.remove(filename)

        return Response(
            stream_with_context(generate()),
            content_type='text/plain; charset=utf-8'
        )
    
    return jsonify({'error': '不支持的文件格式'}), 400

# 会话管理API
@app.route('/api/new-session', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "INSERT INTO sessions (id, title) VALUES (%s, %s)",
            (session_id, '新的对话')
        )
        conn.commit()
        
        return jsonify({
            'id': session_id,
            'title': '新的对话',
            'messages': []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/chat-history')
def get_chat_history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            SELECT s.*, 
                   COUNT(m.id) as message_count,
                   MAX(m.created_at) as last_message_time
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id
            ORDER BY last_message_time DESC NULLS LAST
        """)
        sessions = cursor.fetchall()
        
        return jsonify(sessions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/chat-session/<session_id>')
def get_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 获取会话信息
        cursor.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            return jsonify({'error': '会话不存在'}), 404
        
        # 获取会话的所有消息
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = %s ORDER BY created_at",
            (session_id,)
        )
        messages = cursor.fetchall()
        
        session['messages'] = messages
        return jsonify(session)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/save-message', methods=['POST'])
def save_message():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            """INSERT INTO messages (session_id, content, is_user)
               VALUES (%s, %s, %s)""",
            (data['sessionId'], data['content'], data['isUser'])
        )
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/update-session-title', methods=['POST'])
def update_session_title():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "UPDATE sessions SET title = %s WHERE id = %s",
            (data['title'], data['sessionId'])
        )
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/delete-session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 首先删除会话的所有消息
        cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
        # 然后删除会话
        cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
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

# 用户登录
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    print(f"登录请求数据: {data}")  # 添加调试日志
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (data['username'],)
        )
        user = cursor.fetchone()
        print(f"查询到的用户: {user}")  # 添加调试日志
        
        if not user or not check_password_hash(user['password'], data['password']):
            return jsonify({'error': '用户名或密码错误'}), 401
            
        # 生成 JWT token
        token = jwt.encode({
            'user_id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(days=1)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        print(f"生成的token: {token}")  # 添加调试日志
        
        return jsonify({
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        })
    except Exception as e:
        print(f"登录错误: {str(e)}")  # 添加调试日志
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 用户注销
@app.route('/api/logout', methods=['POST'])
def logout():
    # 前端需要清除 token
    return jsonify({'message': '注销成功'})

# 获取用户信息
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
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify(user)
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'token已过期'}), 401
    except jwt.InvalidTokenError:
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
        
        return jsonify({'message': '收藏成功'})
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
def get_current_user():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': '未登录'}), 401
        
    try:
        # 从 Bearer token 中提取 JWT
        token = token.split(' ')[1]
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT id, username, email, role, created_at, updated_at 
               FROM users 
               WHERE id = %s""",
            (payload['user_id'],)
        )
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify(user)
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'token已过期'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': '无效的token'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()
        
# 更新用户角色（仅管理员可访问）
@app.route('/api/users/<user_id>/role', methods=['PUT'])
@admin_required
def update_user_role(user_id):
    data = request.json
    new_role = data.get('role')
    
    if new_role not in ['user', 'admin']:
        return jsonify({'error': '无效的角色'}), 400
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 检查用户是否存在
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        # 更新用户角色
        cursor.execute(
            "UPDATE users SET role = %s WHERE id = %s",
            (new_role, user_id)
        )
        conn.commit()
        
        return jsonify({'message': '用户角色更新成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 管理员用户管理页面
@app.route('/admin/users')
@admin_required
def admin_users():
    return render_template('admin/users.html')
# 获取所有用户列表（仅管理员可访问）
@app.route('/api/users')
@admin_required
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            """SELECT id, username, email, role, created_at, updated_at 
               FROM users 
               ORDER BY created_at DESC"""
        )
        users = cursor.fetchall()
        
        # 转换datetime对象为字符串
        for user in users:
            if user['created_at']:
                user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if user['updated_at']:
                user['updated_at'] = user['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(users)
    except Exception as e:
        print(f"获取用户列表错误: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
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

# 在应用启动时初始化管理员账户
if __name__ == '__main__':
    init_admin()  # 添加这行
    app.run(debug=True, port=5000)

