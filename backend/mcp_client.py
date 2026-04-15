"""
MCP 客户端 - 真正的 Model Context Protocol 实现
通过 stdio 与 MCP 服务器进程通信
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import time

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    Tool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-client")


class MCPConnection:
    """单个 MCP 服务器连接管理"""
    
    def __init__(self, name: str, server_script: str):
        self.name = name
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._connected = False
    
    async def connect(self) -> bool:
        """连接到 MCP 服务器"""
        try:
            logger.info(f"启动 {self.name} MCP 服务器...")
            
            # 启动服务器进程
            self.process = subprocess.Popen(
                [sys.executable, self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # 创建 stdio 传输流
            self._read_stream = asyncio.StreamReader()
            self._write_protocol = asyncio.Protocol()
            
            # 使用 asyncio.create_subprocess_exec 替代
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                self.server_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 创建 MCP session
            self.session = ClientSession(
                read_stream=self.process.stdout,
                write_stream=self.process.stdin
            )
            
            # 初始化会话
            await self.session.initialize()
            
            # 获取可用工具列表
            tools_response = await self.session.list_tools()
            tool_names = [tool.name for tool in tools_response.tools]
            logger.info(f"{self.name} MCP 服务器已连接，可用工具：{tool_names}")
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"连接 {self.name} MCP 服务器失败：{e}")
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """断开连接并清理资源"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                
                self.process = None
            
            self._connected = False
            logger.info(f"{self.name} MCP 服务器已断开")
            
        except Exception as e:
            logger.error(f"断开 {self.name} MCP 服务器时出错：{e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP 工具"""
        if not self._connected or not self.session:
            raise RuntimeError(f"{self.name} MCP 服务器未连接")
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # 解析结果
            if result.content and len(result.content) > 0:
                # 提取文本内容
                text_content = result.content[0].text
                return json.loads(text_content)
            else:
                return {"success": False, "error": "无返回内容"}
                
        except Exception as e:
            logger.error(f"调用 {self.name}.{tool_name} 失败：{e}")
            return {"success": False, "error": str(e)}
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self.process and self.process.returncode is None


class EnhancedMCPClient:
    """增强的 MCP 客户端，管理多个 MCP 服务器连接"""
    
    def __init__(self, database_mcp_path: str = None, visualization_mcp_path: str = None):
        self.database_mcp_path = database_mcp_path or "mcp_servers/database_mcp/server.py"
        self.visualization_mcp_path = visualization_mcp_path or "mcp_servers/visualization_mcp/server.py"
        
        self.database_connection: Optional[MCPConnection] = None
        self.visualization_connection: Optional[MCPConnection] = None
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化所有 MCP 连接"""
        logger.info("初始化 MCP 客户端...")
        
        try:
            # 连接数据库 MCP
            self.database_connection = MCPConnection("database", self.database_mcp_path)
            db_connected = await self.database_connection.connect()
            
            if not db_connected:
                logger.error("数据库 MCP 连接失败")
                return False
            
            # 连接可视化 MCP
            self.visualization_connection = MCPConnection("visualization", self.visualization_mcp_path)
            viz_connected = await self.visualization_connection.connect()
            
            if not viz_connected:
                logger.warning("可视化 MCP 连接失败，但将继续运行")
            
            self._initialized = True
            logger.info("MCP 客户端初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"MCP 客户端初始化失败：{e}")
            return False
    
    async def shutdown(self):
        """关闭所有连接"""
        logger.info("关闭 MCP 客户端...")
        
        if self.database_connection:
            await self.database_connection.disconnect()
        
        if self.visualization_connection:
            await self.visualization_connection.disconnect()
        
        self._initialized = False
        logger.info("MCP 客户端已关闭")
    
    # ==================== 数据库工具 ====================
    
    async def execute_sql_query(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """执行 SQL 查询（通过 MCP 协议）"""
        if not self._initialized or not self.database_connection:
            return {"success": False, "error": "MCP 客户端未初始化"}
        
        logger.info(f"执行 SQL 查询：{query[:100]}...")
        return await self.database_connection.call_tool(
            "execute_sql_query",
            {"query": query, "limit": limit}
        )
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """获取表结构（通过 MCP 协议）"""
        if not self._initialized or not self.database_connection:
            return {"success": False, "error": "MCP 客户端未初始化"}
        
        logger.info(f"获取表结构：{table_name}")
        return await self.database_connection.call_tool(
            "get_table_schema",
            {"table_name": table_name}
        )
    
    async def list_tables(self) -> Dict[str, Any]:
        """列出所有表（通过 MCP 协议）"""
        if not self._initialized or not self.database_connection:
            return {"success": False, "error": "MCP 客户端未初始化"}
        
        logger.info("列出所有表")
        return await self.database_connection.call_tool("list_tables", {})
    
    # ==================== 可视化工具 ====================
    
    async def create_chart(self, chart_type: str, data: List[Dict], 
                          x_column: str = None, y_column: str = None,
                          title: str = "数据可视化", color_column: str = None,
                          width: int = 800, height: int = 600) -> Dict[str, Any]:
        """创建图表（通过 MCP 协议）"""
        if not self._initialized or not self.visualization_connection:
            return {"success": False, "error": "可视化 MCP 未连接"}
        
        logger.info(f"创建图表：{chart_type}")
        return await self.visualization_connection.call_tool(
            "create_chart",
            {
                "chart_type": chart_type,
                "data": data,
                "x_column": x_column,
                "y_column": y_column,
                "title": title,
                "color_column": color_column,
                "width": width,
                "height": height
            }
        )
    
    async def get_chart_recommendations(self, data: List[Dict]) -> Dict[str, Any]:
        """获取图表推荐（通过 MCP 协议）"""
        if not self._initialized or not self.visualization_connection:
            return {"success": False, "error": "可视化 MCP 未连接"}
        
        logger.info("获取图表推荐")
        return await self.visualization_connection.call_tool(
            "get_chart_recommendations",
            {"data": data}
        )


# 全局客户端实例
mcp_client = EnhancedMCPClient()


@asynccontextmanager
async def get_mcp_client():
    """异步上下文管理器获取 MCP 客户端"""
    client = EnhancedMCPClient()
    try:
        await client.initialize()
        yield client
    finally:
        await client.shutdown()
