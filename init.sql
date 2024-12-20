-- 创建数据库
CREATE DATABASE IF NOT EXISTS doc_summary CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE doc_summary;

-- 设置外键检查为0
SET FOREIGN_KEY_CHECKS=0;

-- 删除已存在的表
DROP TABLE IF EXISTS document_summaries;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS file_mappings;

-- 创建用户表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(120) NOT NULL UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建文档摘要表
CREATE TABLE document_summaries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_hash VARCHAR(32) NOT NULL,
    summary_text LONGTEXT NOT NULL,
    original_text LONGTEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME ON UPDATE CURRENT_TIMESTAMP,
    summary_length VARCHAR(20),
    target_language VARCHAR(20),
    file_content LONGBLOB,
    file_size BIGINT,
    mime_type VARCHAR(100),
    original_filename VARCHAR(255),
    display_filename VARCHAR(255),
    INDEX idx_file_hash (file_hash),
    INDEX idx_created_at (created_at)
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

-- 创建管理员账户
INSERT INTO users (username, password, email, role) 
VALUES (
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewFpxQgkD8ECwmy.',
    'admin@example.com',
    'admin'
) ON DUPLICATE KEY UPDATE username=username;

-- 设置外键检查为1
SET FOREIGN_KEY_CHECKS=1;