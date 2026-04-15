"""
项目配置管理
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """应用配置"""
    
    # vLLM 配置
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "vllm_api_key")
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    
    # MCP 服务器配置
    DATABASE_MCP_PORT = int(os.getenv("DATABASE_MCP_PORT", "8081"))
    VISUALIZATION_MCP_PORT = int(os.getenv("VISUALIZATION_MCP_PORT", "8082"))
    
    # 数据库配置 (只读用户)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "readonly_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "readonly_password")
    DB_NAME = os.getenv("DB_NAME", "doc_summary")
    
    # Flask 配置
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # Redis 配置 (可选，用于缓存)
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 查询限制
    MAX_QUERY_ROWS = int(os.getenv("MAX_QUERY_ROWS", "1000"))
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))
    
    # 图表配置
    DEFAULT_CHART_WIDTH = int(os.getenv("DEFAULT_CHART_WIDTH", "800"))
    DEFAULT_CHART_HEIGHT = int(os.getenv("DEFAULT_CHART_HEIGHT", "600"))


settings = Settings()
