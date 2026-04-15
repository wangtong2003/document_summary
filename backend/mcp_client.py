"""
MCP 客户端 - 连接 MCP 服务器
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-client")


class MCPClient:
    """MCP 客户端，管理多个 MCP 服务器连接"""
    
    def __init__(self, database_mcp_path: str = None, visualization_mcp_path: str = None):
        self.database_mcp_path = database_mcp_path or "mcp_servers/database_mcp/server.py"
        self.visualization_mcp_path = visualization_mcp_path or "mcp_servers/visualization_mcp/server.py"
        
        self.database_session: Optional[ClientSession] = None
        self.visualization_session: Optional[ClientSession] = None
        
        self.database_process: Optional[subprocess.Popen] = None
        self.visualization_process: Optional[subprocess.Popen] = None
    
    async def start_database_mcp(self):
        """启动数据库 MCP 服务器"""
        logger.info("启动数据库 MCP 服务器...")
        
        try:
            # 启动 MCP 服务器进程
            self.database_process = subprocess.Popen(
                [sys.executable, self.database_mcp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 创建 stdio 传输
            read_stream = asyncio.StreamReader()
            write_protocol = asyncio.Protocol()
            
            # 这里需要实现完整的 MCP 客户端协议
            # 简化版本：直接调用工具函数
            logger.info("数据库 MCP 服务器已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动数据库 MCP 失败：{e}")
            return False
    
    async def start_visualization_mcp(self):
        """启动可视化 MCP 服务器"""
        logger.info("启动可视化 MCP 服务器...")
        
        try:
            self.visualization_process = subprocess.Popen(
                [sys.executable, self.visualization_mcp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("可视化 MCP 服务器已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动可视化 MCP 失败：{e}")
            return False
    
    async def stop(self):
        """停止所有 MCP 服务器"""
        logger.info("停止 MCP 服务器...")
        
        if self.database_process:
            self.database_process.terminate()
            self.database_process.wait()
        
        if self.visualization_process:
            self.visualization_process.terminate()
            self.visualization_process.wait()
        
        logger.info("所有 MCP 服务器已停止")
    
    # 数据库工具调用 (简化版本，实际应通过 MCP 协议)
    async def execute_sql_query(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """执行 SQL 查询"""
        logger.info(f"执行 SQL 查询：{query[:100]}...")
        
        # TODO: 通过 MCP 协议调用数据库 MCP 服务器
        # 这里使用直接导入的方式作为临时方案
        try:
            from mcp_servers.database_mcp.server import DB_CONFIG, validate_query
            import pymysql
            from pymysql.cursors import DictCursor
            
            # 验证查询
            if not validate_query(query):
                return {
                    "success": False,
                    "error": "不允许的 SQL 语句，仅支持 SELECT 查询"
                }
            
            # 添加 LIMIT
            if 'LIMIT' not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            # 执行查询
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                result_list = [dict(row) for row in results]
                
                return {
                    "success": True,
                    "data": result_list,
                    "row_count": len(result_list)
                }
        
        except Exception as e:
            logger.error(f"SQL 查询失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """获取表结构"""
        logger.info(f"获取表结构：{table_name}")
        
        try:
            from mcp_servers.database_mcp.server import DB_CONFIG
            import pymysql
            from pymysql.cursors import DictCursor
            
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE `{table_name}`")
                schema = cursor.fetchall()
                
                return {
                    "success": True,
                    "schema": [dict(row) for row in schema],
                    "table": table_name
                }
        
        except Exception as e:
            logger.error(f"获取表结构失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_tables(self) -> Dict[str, Any]:
        """列出所有表"""
        logger.info("列出所有表")
        
        try:
            from mcp_servers.database_mcp.server import DB_CONFIG
            import pymysql
            
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [row[f'Tables_in_{DB_CONFIG["database"]}'] for row in cursor.fetchall()]
                
                return {
                    "success": True,
                    "tables": tables
                }
        
        except Exception as e:
            logger.error(f"列出表失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # 可视化工具调用
    async def create_chart(self, chart_type: str, data: List[Dict], 
                          x_column: str = None, y_column: str = None,
                          title: str = "数据可视化", color_column: str = None,
                          width: int = 800, height: int = 600) -> Dict[str, Any]:
        """创建图表"""
        logger.info(f"创建图表：{chart_type}")
        
        try:
            from mcp_servers.visualization_mcp.server import create_plotly_chart
            
            html_content = create_plotly_chart(
                chart_type=chart_type,
                data=data,
                x_column=x_column,
                y_column=y_column,
                title=title,
                color_column=color_column,
                width=width,
                height=height
            )
            
            return {
                "success": True,
                "chart_html": html_content,
                "chart_type": chart_type
            }
        
        except Exception as e:
            logger.error(f"创建图表失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_chart_recommendations(self, data: List[Dict]) -> Dict[str, Any]:
        """获取图表推荐"""
        logger.info("获取图表推荐")
        
        try:
            from mcp_servers.visualization_mcp.server import app as viz_app
            import asyncio
            
            # 调用工具
            result = await viz_app.call_tool(
                "get_chart_recommendations",
                {"data": data}
            )
            
            if result and len(result) > 0:
                response_data = json.loads(result[0].text)
                return response_data
            
            return {
                "success": False,
                "error": "无法获取图表推荐"
            }
        
        except Exception as e:
            logger.error(f"获取图表推荐失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }


# 全局客户端实例
mcp_client = MCPClient()
