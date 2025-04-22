-- 创建数据库
CREATE DATABASE IF NOT EXISTS doc_summary CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE doc_summary;

-- 设置外键检查为0
SET FOREIGN_KEY_CHECKS=0;

-- 删除已存在的表
DROP TABLE IF EXISTS file_chunks;
DROP TABLE IF EXISTS file_mappings;
DROP TABLE IF EXISTS summaries;
DROP TABLE IF EXISTS document_summaries;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS system_settings;

-- 创建用户表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID，自增主键',
    username VARCHAR(80) NOT NULL UNIQUE COMMENT '用户名，唯一',
    password VARCHAR(255) NOT NULL COMMENT '密码，加密存储',
    email VARCHAR(120) NOT NULL UNIQUE COMMENT '电子邮箱，唯一',
    role VARCHAR(20) NOT NULL DEFAULT 'user' COMMENT '用户角色，默认为普通用户',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_username (username) COMMENT '用户名索引',
    INDEX idx_role (role) COMMENT '角色索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '用户信息表';

-- 创建文档摘要表
CREATE TABLE document_summaries (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '摘要ID，自增主键',
    file_name VARCHAR(255) NOT NULL COMMENT '文件名',
    file_hash VARCHAR(32) COMMENT '文件哈希值，用于去重',
    original_text MEDIUMTEXT COMMENT '原始文本内容',
    summary_text MEDIUMTEXT COMMENT '摘要内容',
    file_content MEDIUMBLOB COMMENT '文件二进制内容（小文件直接存储）',
    content_vectors MEDIUMBLOB COMMENT '内容的向量数据',
    summary_vectors MEDIUMBLOB COMMENT '摘要的向量数据',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    summary_length VARCHAR(20) COMMENT '摘要长度设置（short/medium/long）',
    target_language VARCHAR(20) COMMENT '目标语言',
    file_size BIGINT COMMENT '文件大小（字节）',
    mime_type VARCHAR(100) COMMENT '文件MIME类型',
    original_filename VARCHAR(255) COMMENT '原始文件名',
    display_filename VARCHAR(255) COMMENT '显示用文件名',
    keywords VARCHAR(255) COMMENT '关键词，以分隔符分隔',
    topic_analysis JSON COMMENT '主题分析结果，JSON格式',
    embedding_model VARCHAR(100) COMMENT '使用的嵌入模型名称',
    chunks_info JSON COMMENT '分块信息，JSON格式',
    is_chunked BOOLEAN DEFAULT FALSE COMMENT '是否已分块存储',
    total_chunks INT DEFAULT 0 COMMENT '总分块数',
    chroma_collection VARCHAR(100) COMMENT 'Chroma向量数据库集合名称',
    has_vector_store BOOLEAN DEFAULT FALSE COMMENT '是否已创建向量存储',
    user_id INT NOT NULL COMMENT '所属用户ID', 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_file_hash (file_hash) COMMENT '文件哈希索引',
    INDEX idx_created_at (created_at) COMMENT '创建时间索引',
    INDEX idx_user_id (user_id) COMMENT '用户ID索引',
    FULLTEXT INDEX ft_keywords (keywords) COMMENT '关键词全文索引',
    FULLTEXT INDEX ft_summary (summary_text(512)) COMMENT '摘要内容全文索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '文档摘要表';

-- 创建文件块表
CREATE TABLE file_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '块ID，自增主键',
    document_id INT NOT NULL COMMENT '所属文档ID',
    chunk_index INT NOT NULL COMMENT '块索引号',
    chunk_data MEDIUMBLOB NOT NULL COMMENT '块数据内容，最大10MB',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (document_id) REFERENCES document_summaries(id) ON DELETE CASCADE,
    INDEX idx_document_chunk (document_id, chunk_index) COMMENT '文档ID和块索引的复合索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '文件分块存储表';

-- 创建文件名映射表
CREATE TABLE file_mappings (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '映射ID，自增主键',
    summary_id INT NOT NULL COMMENT '所属摘要ID',
    original_filename VARCHAR(255) NOT NULL COMMENT '原始文件名',
    system_filename VARCHAR(255) NOT NULL COMMENT '系统存储的文件名',
    display_filename VARCHAR(255) NOT NULL COMMENT '显示用文件名',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (summary_id) REFERENCES document_summaries(id) ON DELETE CASCADE,
    INDEX idx_summary_id (summary_id) COMMENT '摘要ID索引',
    INDEX idx_system_filename (system_filename) COMMENT '系统文件名索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '文件名映射表';

-- 创建文本摘要表
CREATE TABLE summaries (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '摘要ID，自增主键',
    original_text MEDIUMTEXT NOT NULL COMMENT '原始文本内容',
    summary_text MEDIUMTEXT NOT NULL COMMENT '摘要内容',
    keywords VARCHAR(255) COMMENT '关键词，以分隔符分隔',
    model VARCHAR(100) COMMENT '使用的模型名称',
    target_language VARCHAR(20) COMMENT '目标语言',
    target_length VARCHAR(20) COMMENT '目标长度设置',
    focus_areas VARCHAR(255) COMMENT '重点关注领域',
    level VARCHAR(20) COMMENT '专业程度',
    language_style VARCHAR(20) COMMENT '语言风格',
    style VARCHAR(20) COMMENT '写作风格',
    format VARCHAR(20) COMMENT '输出格式',
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_timestamp (timestamp) COMMENT '时间戳索引',
    FULLTEXT INDEX ft_keywords (keywords) COMMENT '关键词全文索引',
    FULLTEXT INDEX ft_summary (summary_text(512)) COMMENT '摘要内容全文索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '文本摘要表';

-- 创建会话表
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY COMMENT '会话ID',
    user_id INT COMMENT '用户ID',
    data BLOB NOT NULL COMMENT '会话数据',
    expiry DATETIME NOT NULL COMMENT '过期时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_expiry (expiry) COMMENT '过期时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '用户会话表';

-- 创建系统设置表
CREATE TABLE system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '设置ID，自增主键',
    setting_key VARCHAR(100) NOT NULL UNIQUE COMMENT '设置键名',
    setting_value TEXT NOT NULL COMMENT '设置值',
    setting_type VARCHAR(50) NOT NULL COMMENT '设置类型',
    description TEXT COMMENT '设置描述',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_setting_key (setting_key) COMMENT '设置键名索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT '系统设置表';

-- 创建管理员账户 (密码: admin123)
INSERT INTO users (username, password, email, role) 
VALUES (
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewFpxQgkD8ECwmy.',
    'admin@example.com',
    'admin'
) ON DUPLICATE KEY UPDATE username=username;

-- 插入系统默认设置
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('default_summary_length', 'medium', 'string', '默认摘要长度'),
('default_target_language', 'chinese', 'string', '默认目标语言'),
('embedding_model', 'snowflake-arctic-embed2', 'string', '默认向量嵌入模型'),
('max_file_size', '50', 'integer', '最大文件大小(MB)'),
('allowed_file_types', 'pdf,docx,txt,md,epub', 'string', '允许上传的文件类型');

-- 设置外键检查为1
SET FOREIGN_KEY_CHECKS=1;