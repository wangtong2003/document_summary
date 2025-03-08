-- 创建数据库
CREATE DATABASE IF NOT EXISTS doc_summary CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE doc_summary;

-- 设置外键检查为0
SET FOREIGN_KEY_CHECKS=0;

-- 删除已存在的表
DROP TABLE IF EXISTS file_chunks;
DROP TABLE IF EXISTS file_mappings;
DROP TABLE IF EXISTS document_summaries;
DROP TABLE IF EXISTS users;

-- 创建用户表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(120) NOT NULL UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建文档摘要表
CREATE TABLE document_summaries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_hash VARCHAR(32),
    original_text MEDIUMTEXT,
    summary_text MEDIUMTEXT,
    file_content MEDIUMBLOB,
    content_vectors MEDIUMBLOB,
    summary_vectors MEDIUMBLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    summary_length VARCHAR(20),
    target_language VARCHAR(20),
    file_size BIGINT,
    mime_type VARCHAR(100),
    original_filename VARCHAR(255),
    display_filename VARCHAR(255),
    keywords VARCHAR(255),
    topic_analysis JSON,
    embedding_model VARCHAR(100),
    chunks_info JSON,
    is_chunked BOOLEAN DEFAULT FALSE,
    total_chunks INT DEFAULT 0,
    user_id INT, 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_file_hash (file_hash),
    INDEX idx_created_at (created_at),
    INDEX idx_user_id (user_id),
    FULLTEXT INDEX ft_keywords (keywords),
    FULLTEXT INDEX ft_summary (summary_text(512))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建文件块表
CREATE TABLE file_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_data MEDIUMBLOB NOT NULL,  -- 10MB per chunk as defined in model
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES document_summaries(id) ON DELETE CASCADE,
    INDEX idx_document_chunk (document_id, chunk_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建文件名映射表
CREATE TABLE file_mappings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    summary_id INT NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    system_filename VARCHAR(255) NOT NULL,
    display_filename VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (summary_id) REFERENCES document_summaries(id) ON DELETE CASCADE,
    INDEX idx_summary_id (summary_id),
    INDEX idx_system_filename (system_filename)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建搜索历史表
CREATE TABLE search_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    document_id INT NOT NULL,
    query_text VARCHAR(255) NOT NULL,
    search_type VARCHAR(50) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES document_summaries(id) ON DELETE CASCADE,
    INDEX idx_user_document (user_id, document_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建会话表
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INT,
    data BLOB NOT NULL,
    expiry DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_expiry (expiry)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建系统设置表
CREATE TABLE system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(50) NOT NULL,
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_setting_key (setting_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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