"""
智能问数系统 - FastAPI 后端主应用
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from langgraph_agent.graph import build_agent_graph, run_agent
from backend.mcp_client import mcp_client
from config.settings import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smart-analytics")

app = FastAPI(
    title="智能问数系统 API",
    description="基于 LangGraph + MCP + vLLM 的智能数据分析与可视化系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str

# HTML 前端模板 (简化版本，生产环境应使用独立前端项目)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能问数系统</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: white; margin-bottom: 30px; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .input-section { background: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); }
        textarea { width: 100%; height: 120px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 15px; resize: vertical; transition: border-color 0.3s; }
        textarea:focus { outline: none; border-color: #667eea; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 40px; border-radius: 8px; cursor: pointer; font-size: 16px; margin-top: 15px; transition: transform 0.2s, box-shadow 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        button:disabled { background: #ccc; cursor: not-allowed; transform: none; box-shadow: none; }
        .result-section { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); display: none; }
        .response-text { margin-bottom: 25px; line-height: 1.8; color: #333; font-size: 15px; }
        .chart-container { margin-top: 25px; border-radius: 8px; overflow: hidden; }
        .loading { text-align: center; padding: 50px; display: none; background: white; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { color: #dc3545; padding: 15px; background: #f8d7da; border-radius: 8px; margin-top: 15px; border-left: 4px solid #dc3545; }
        .sql-display { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; font-family: 'Courier New', monospace; font-size: 13px; overflow-x: auto; border-left: 4px solid #667eea; }
        .section-title { color: #667eea; margin-bottom: 15px; font-size: 18px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 智能问数系统</h1>
        <div class="input-section">
            <textarea id="queryInput" placeholder="请输入您的问题，例如：&#10;- 统计每个用户的文档数量&#10;- 显示最近创建的 10 个文档&#10;- 分析文档摘要的分布情况"></textarea>
            <button onclick="submitQuery()" id="submitBtn">提交查询</button>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 20px; color: #666; font-size: 16px;">正在处理您的查询，请稍候...</p>
        </div>
        <div class="result-section" id="resultSection">
            <div class="section-title">📝 分析结果</div>
            <div class="response-text" id="responseText"></div>
            
            <div id="sqlSection" style="display:none;">
                <div class="section-title">💾 SQL 查询</div>
                <div class="sql-display" id="sqlQuery"></div>
            </div>
            
            <div class="section-title" style="margin-top: 25px;">📈 可视化图表</div>
            <div class="chart-container" id="chartContainer">
                <p style="color: #999; text-align: center; padding: 40px;">暂无图表数据</p>
            </div>
        </div>
    </div>
    <script>
        async function submitQuery() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) { alert('请输入查询内容'); return; }
            
            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const resultSection = document.getElementById('resultSection');
            const sqlSection = document.getElementById('sqlSection');
            
            btn.disabled = true;
            loading.style.display = 'block';
            resultSection.style.display = 'none';
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                const endTime = Date.now();
                const executionTime = ((endTime - startTime) / 1000).toFixed(2);
                
                if (data.success) {
                    let responseHtml = data.response.replace(/\\n/g, '<br>');
                    if (data.execution_time) {
                        responseHtml += `<br><br><small style="color:#999;">⏱️ 执行时间：${data.execution_time}秒</small>`;
                    }
                    
                    document.getElementById('responseText').innerHTML = responseHtml;
                    
                    if (data.sql_query) {
                        document.getElementById('sqlQuery').textContent = data.sql_query;
                        sqlSection.style.display = 'block';
                    } else {
                        sqlSection.style.display = 'none';
                    }
                    
                    if (data.chart_html) {
                        document.getElementById('chartContainer').innerHTML = data.chart_html;
                    } else if (data.chart_config) {
                        Plotly.newPlot('chartContainer', data.chart_config.data, data.chart_config.layout);
                    } else {
                        document.getElementById('chartContainer').innerHTML = '<p style="color: #999; text-align: center; padding: 40px;">本次查询未生成图表</p>';
                    }
                    
                    resultSection.style.display = 'block';
                } else {
                    alert('查询失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                submitQuery();
            }
        });
    </script>
</body>
</html>
"""


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
        
        # 运行 LangGraph Agent
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
            success=True,
            response=final_response,
            chart_html=chart_html,
            chart_config=chart_config,
            sql_query=result.get('generated_sql', ''),
            data_summary=result.get('data_summary', {}),
            execution_time=execution_time
        )
    
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
        result = await mcp_client.list_tables()
        return result
    except Exception as e:
        logger.error(f"列出表失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schema/{table_name}")
async def get_table_schema(table_name: str):
    """获取表结构"""
    try:
        result = await mcp_client.get_table_schema(table_name)
        return result
    except Exception as e:
        logger.error(f"获取表结构失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        service="smart-analytics",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


if __name__ == '__main__':
    import uvicorn
    
    logger.info("启动智能问数系统...")
    logger.info(f"vLLM 服务地址：{settings.VLLM_BASE_URL}")
    logger.info(f"数据库：{settings.DB_NAME}@{settings.DB_HOST}:{settings.DB_PORT}")
    
    # 启动 FastAPI 应用
    uvicorn.run(
        app,
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
