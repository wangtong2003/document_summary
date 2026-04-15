"""
智能问数系统 - FastAPI 后端主应用
集成 MCP、缓存、连接池和监控
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from langgraph_agent.runner import init_agent, run_agent, shutdown_agent
from backend.cache_manager import get_cache_manager
from config.settings import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smart-analytics")

# 请求/响应模型
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询问题")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "统计每个用户的文档数量，并按降序排列"
            }
        }

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    chart_html: Optional[str] = None
    chart_config: Optional[Dict[str, Any]] = None
    sql_query: Optional[str] = None
    data_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    from_cache: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str
    mcp_connected: bool = False
    cache_enabled: bool = False

class MetricsResponse(BaseModel):
    cache_stats: Optional[Dict[str, Any]] = None
    uptime_seconds: float = 0.0


# 全局启动时间
start_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("启动智能问数系统...")
    logger.info(f"vLLM 服务地址：{settings.VLLM_BASE_URL}")
    logger.info(f"数据库：{settings.DB_NAME}@{settings.DB_HOST}:{settings.DB_PORT}")
    
    # 初始化 Agent（包含 MCP 客户端）
    agent_initialized = await init_agent()
    if not agent_initialized:
        logger.error("Agent 初始化失败，但将继续运行")
    
    yield
    
    # 关闭时清理资源
    logger.info("关闭智能问数系统...")
    await shutdown_agent()


app = FastAPI(
    title="智能问数系统 API",
    description="基于 LangGraph + MCP + vLLM 的智能数据分析与可视化系统",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载前端静态文件
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# P1: 错误分类和处理
class SmartAnalyticsError(Exception):
    """自定义异常基类"""
    pass

class MCPConnectionError(SmartAnalyticsError):
    """MCP 连接错误"""
    pass

class DatabaseQueryError(SmartAnalyticsError):
    """数据库查询错误"""
    pass

class LLMTimeoutError(SmartAnalyticsError):
    """LLM 超时错误"""
    pass


@app.exception_handler(SmartAnalyticsError)
async def smart_error_handler(request: Request, exc: SmartAnalyticsError):
    """统一错误处理"""
    error_type = type(exc).__name__
    logger.error(f"{error_type}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": error_type
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常：{exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "服务器内部错误",
            "error_type": "InternalError"
        }
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 重定向到前端"""
    from fastapi.responses import FileResponse
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="Frontend not found", status_code=404)


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """处理用户查询"""
    start_time = time.time()
    
    try:
        logger.info(f"收到查询：{req.query}")
        
        # 运行 Agent
        result = await run_agent(req.query)
        
        execution_time = time.time() - start_time
        
        # 提取结果
        final_response = result.get('final_response', '')
        chart_html = result.get('chart_html', None)
        chart_config = result.get('chart_config', None)
        query_error = result.get('query_error', None)
        
        if query_error:
            return QueryResponse(
                success=False,
                error=query_error,
                execution_time=execution_time
            )
        
        return QueryResponse(
            success=result.get('success', False),
            response=final_response,
            chart_html=chart_html,
            chart_config=chart_config,
            sql_query=result.get('generated_sql', ''),
            data_summary=result.get('data_summary', {}),
            execution_time=execution_time,
            from_cache=result.get('from_cache', False)
        )
    
    except asyncio.TimeoutError as e:
        logger.error(f"查询超时：{e}")
        raise LLMTimeoutError("查询处理超时，请稍后重试")
    
    except Exception as e:
        logger.error(f"处理查询失败：{e}", exc_info=True)
        return QueryResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )


@app.get("/api/tables")
async def list_tables():
    """列出所有数据库表"""
    try:
        from backend.mcp_client import mcp_client
        result = await mcp_client.list_tables()
        return result
    except Exception as e:
        logger.error(f"列出表失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schema/{table_name}")
async def get_table_schema(table_name: str):
    """获取表结构"""
    try:
        from backend.mcp_client import mcp_client
        result = await mcp_client.get_table_schema(table_name)
        return result
    except Exception as e:
        logger.error(f"获取表结构失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    from backend.mcp_client import mcp_client
    
    return HealthResponse(
        status="healthy",
        service="smart-analytics",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        mcp_connected=mcp_client._initialized,
        cache_enabled=get_cache_manager()._initialized
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """获取系统指标 (P2: 监控指标)"""
    from backend.metrics import get_metrics_collector
    
    cache = get_cache_manager()
    cache_stats = await cache.get_stats()
    
    # 获取 Prometheus 指标
    metrics = await get_metrics_collector().get_metrics()
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return MetricsResponse(
        cache_stats=cache_stats,
        uptime_seconds=uptime
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 格式的监控指标"""
    from backend.metrics import get_metrics_collector
    from fastapi.responses import PlainTextResponse
    
    collector = get_metrics_collector()
    return PlainTextResponse(collector.to_prometheus_format())


@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    try:
        cache = get_cache_manager()
        success = await cache.clear_all()
        return {"success": success, "message": "缓存已清空" if success else "清空失败"}
    except Exception as e:
        logger.error(f"清空缓存失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    logger.info("启动智能问数系统...")
    logger.info(f"vLLM 服务地址：{settings.VLLM_BASE_URL}")
    logger.info(f"数据库：{settings.DB_NAME}@{settings.DB_HOST}:{settings.DB_PORT}")
    
    # 启动 FastAPI 应用 - 生产环境优化配置
    uvicorn.run(
        app,
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        workers=4,              # 多进程工作模式
        loop="uvloop",          # 高性能事件循环
        http="httptools",       # 高性能 HTTP 解析
        ws="websockets",        # WebSocket 支持
        timeout_keep_alive=30,  # 保持连接超时时间
        limit_concurrency=100,  # 最大并发连接数
        backlog=2048,           # 监听队列大小
        access_log=True         # 启用访问日志
    )
