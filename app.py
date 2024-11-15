from flask import Flask, request, jsonify, Response, render_template, stream_with_context
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
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'epub'}

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

# 路由
@app.route('/')
def index():
    return render_template('index.html')

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
                    model='qwen2.5:3b',
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

if __name__ == '__main__':
    app.run(debug=True, port=5000) 