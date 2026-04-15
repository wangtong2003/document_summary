"""
数据库连接池管理模块
使用 SQLAlchemy 实现高效的数据库连接池
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db-pool")


class DatabaseConnectionPool:
    """数据库连接池管理器"""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'readonly_user'),
            'password': os.getenv('DB_PASSWORD', 'readonly_password'),
            'database': os.getenv('DB_NAME', 'doc_summary'),
            'charset': 'utf8mb4'
        }
        
        self.engine = None
        self._initialized = False
    
    def initialize(self, pool_size: int = 10, max_overflow: int = 20, 
                   pool_timeout: int = 30, pool_recycle: int = 3600):
        """初始化数据库连接池"""
        try:
            config = self.db_config
            connection_url = (
                f"mysql+pymysql://{config['user']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}"
                f"?charset={config['charset']}"
            )
            
            self.engine = create_engine(
                connection_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # 自动检测失效连接
                echo=False  # 生产环境关闭 SQL 日志
            )
            
            # 测试连接
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
            
            self._initialized = True
            logger.info(f"数据库连接池初始化成功 (pool_size={pool_size}, max_overflow={max_overflow})")
            return True
            
        except Exception as e:
            logger.error(f"数据库连接池初始化失败：{e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        if not self._initialized:
            raise RuntimeError("数据库连接池未初始化")
        
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            logger.error(f"数据库连接错误：{e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Dict = None, limit: int = 100):
        """执行查询并返回结果"""
        try:
            with self.get_connection() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                columns = [col[0] for col in result.cursor.description]
                rows = result.fetchall()
                
                # 转换为字典列表
                data = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data)
                }
                
        except Exception as e:
            logger.error(f"查询执行失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_table_schema(self, table_name: str):
        """获取表结构"""
        query = f"DESCRIBE `{table_name}`"
        return self.execute_query(query)
    
    def list_tables(self):
        """列出所有表"""
        query = "SHOW TABLES"
        result = self.execute_query(query)
        
        if result.get('success'):
            # 提取表名列表
            db_name = self.db_config['database']
            tables = [row[f'Tables_in_{db_name}'] for row in result.get('data', [])]
            return {
                "success": True,
                "tables": tables
            }
        
        return result
    
    def close(self):
        """关闭连接池"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
            logger.info("数据库连接池已关闭")


# 全局连接池实例
db_pool = DatabaseConnectionPool()


def init_db_pool(db_config: Dict[str, Any] = None, **pool_kwargs):
    """初始化全局数据库连接池"""
    global db_pool
    db_pool = DatabaseConnectionPool(db_config)
    return db_pool.initialize(**pool_kwargs)


def get_db_pool() -> DatabaseConnectionPool:
    """获取全局数据库连接池实例"""
    return db_pool
