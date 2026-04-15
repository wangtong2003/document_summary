"""
数据库 MCP 服务器
提供安全的只读数据库查询接口
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import pymysql
from pymysql.cursors import DictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database-mcp")

# 数据库配置 (应从环境变量读取)
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'readonly_user',
    'password': 'readonly_password',
    'database': 'doc_summary',
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

app = Server("database-mcp")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出可用的数据库工具"""
    return [
        Tool(
            name="execute_sql_query",
            description="执行只读 SQL 查询语句，返回查询结果。仅支持 SELECT 语句。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL 查询语句 (仅限 SELECT)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果的最大行数",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_table_schema",
            description="获取数据库表结构信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表名"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="list_tables",
            description="列出数据库中所有表",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

def validate_query(query: str) -> bool:
    """验证 SQL 查询是否安全 (仅允许 SELECT)"""
    query_upper = query.strip().upper()
    # 只允许 SELECT 开头的查询
    if not query_upper.startswith('SELECT'):
        return False
    # 禁止危险关键字
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False
    return True

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """执行数据库工具调用"""
    connection = None
    try:
        if name == "execute_sql_query":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 100)
            
            # 验证查询安全性
            if not validate_query(query):
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "不允许的 SQL 语句，仅支持 SELECT 查询",
                        "success": False
                    }, ensure_ascii=False)
                )]
            
            # 添加 LIMIT 限制
            if 'LIMIT' not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            # 执行查询
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
                # 转换为 JSON
                result_list = [dict(row) for row in results]
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "data": result_list,
                        "row_count": len(result_list),
                        "success": True
                    }, ensure_ascii=False, default=str)
                )]
        
        elif name == "get_table_schema":
            table_name = arguments.get("table_name", "")
            if not table_name:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "表名不能为空", "success": False})
                )]
            
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE `{table_name}`")
                schema = cursor.fetchall()
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "schema": [dict(row) for row in schema],
                        "table": table_name,
                        "success": True
                    }, ensure_ascii=False, default=str)
                )]
        
        elif name == "list_tables":
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [row[f'Tables_in_{DB_CONFIG["database"]}'] for row in cursor.fetchall()]
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "tables": tables,
                        "success": True
                    }, ensure_ascii=False)
                )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"未知工具：{name}", "success": False})
            )]
    
    except Exception as e:
        logger.error(f"执行工具 {name} 时出错：{e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "success": False
            }, ensure_ascii=False)
        )]
    finally:
        if connection:
            connection.close()

async def main():
    """启动 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
