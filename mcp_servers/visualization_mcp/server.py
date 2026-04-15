"""
可视化 MCP 服务器
基于 Plotly 生成交互式图表
"""

import asyncio
import json
import logging
import base64
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualization-mcp")

app = Server("visualization-mcp")

# 支持的图表类型
CHART_TYPES = {
    "bar": "柱状图",
    "line": "折线图",
    "scatter": "散点图",
    "pie": "饼图",
    "area": "面积图",
    "histogram": "直方图",
    "box": "箱线图",
    "heatmap": "热力图",
    "treemap": "树状图",
    "funnel": "漏斗图"
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出可用的可视化工具"""
    return [
        Tool(
            name="create_chart",
            description=f"根据数据创建可视化图表。支持的图表类型：{', '.join(CHART_TYPES.keys())}",
            inputSchema={
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "description": "图表类型",
                        "enum": list(CHART_TYPES.keys())
                    },
                    "data": {
                        "type": "array",
                        "description": "数据数组，每个元素是一个对象",
                        "items": {"type": "object"}
                    },
                    "x_column": {
                        "type": "string",
                        "description": "X 轴数据列名"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Y 轴数据列名"
                    },
                    "title": {
                        "type": "string",
                        "description": "图表标题"
                    },
                    "color_column": {
                        "type": "string",
                        "description": "颜色分组列名 (可选)"
                    },
                    "width": {
                        "type": "integer",
                        "description": "图表宽度",
                        "default": 800
                    },
                    "height": {
                        "type": "integer",
                        "description": "图表高度",
                        "default": 600
                    }
                },
                "required": ["chart_type", "data"]
            }
        ),
        Tool(
            name="get_chart_recommendations",
            description="根据数据结构推荐合适的图表类型",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "示例数据",
                        "items": {"type": "object"}
                    },
                    "columns": {
                        "type": "array",
                        "description": "列名列表",
                        "items": {"type": "string"}
                    }
                },
                "required": ["data"]
            }
        )
    ]

def create_plotly_chart(
    chart_type: str,
    data: List[Dict],
    x_column: str = None,
    y_column: str = None,
    title: str = "数据可视化",
    color_column: str = None,
    width: int = 800,
    height: int = 600
) -> str:
    """创建 Plotly 图表并返回 HTML"""
    df = pd.DataFrame(data)
    
    # 自动推断列
    if not x_column and len(df.columns) > 0:
        x_column = df.columns[0]
    if not y_column and len(df.columns) > 1:
        y_column = df.columns[1]
    
    fig = None
    
    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, values=y_column, names=x_column, title=title)
        elif chart_type == "area":
            fig = px.area(df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_column or y_column, color=color_column, title=title)
        elif chart_type == "box":
            fig = px.box(df, y=y_column, x=color_column, title=title)
        elif chart_type == "heatmap":
            fig = px.imshow(df.pivot(index=x_column, columns=color_column, values=y_column), 
                          title=title)
        elif chart_type == "treemap":
            fig = px.treemap(df, path=[x_column], values=y_column, title=title)
        elif chart_type == "funnel":
            fig = px.funnel(df, x=y_column, y=x_column, title=title)
        
        if fig is None:
            raise ValueError(f"不支持的图表类型：{chart_type}")
        
        # 更新布局
        fig.update_layout(
            width=width,
            height=height,
            showlegend=True,
            template="plotly_white"
        )
        
        # 转换为 HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        return html
    
    except Exception as e:
        logger.error(f"创建图表失败：{e}")
        raise

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """执行可视化工具调用"""
    try:
        if name == "create_chart":
            chart_type = arguments.get("chart_type", "bar")
            data = arguments.get("data", [])
            x_column = arguments.get("x_column")
            y_column = arguments.get("y_column")
            title = arguments.get("title", "数据可视化")
            color_column = arguments.get("color_column")
            width = arguments.get("width", 800)
            height = arguments.get("height", 600)
            
            if not data:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "数据不能为空", "success": False})
                )]
            
            # 创建图表
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
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "chart_html": html_content,
                    "chart_type": chart_type,
                    "success": True
                }, ensure_ascii=False)
            )]
        
        elif name == "get_chart_recommendations":
            data = arguments.get("data", [])
            columns = arguments.get("columns", [])
            
            if not data:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "数据不能为空", "success": False})
                )]
            
            df = pd.DataFrame(data)
            recommendations = []
            
            # 基于数据特征推荐图表
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                recommendations.append({
                    "type": "scatter",
                    "reason": "适合展示两个数值变量之间的关系",
                    "suggested_x": numeric_cols[0],
                    "suggested_y": numeric_cols[1]
                })
                recommendations.append({
                    "type": "line",
                    "reason": "适合展示时间序列或趋势",
                    "suggested_x": numeric_cols[0],
                    "suggested_y": numeric_cols[1]
                })
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                recommendations.append({
                    "type": "bar",
                    "reason": "适合分类数据对比",
                    "suggested_x": categorical_cols[0],
                    "suggested_y": numeric_cols[0]
                })
                recommendations.append({
                    "type": "pie",
                    "reason": "适合展示占比关系",
                    "suggested_x": categorical_cols[0],
                    "suggested_y": numeric_cols[0]
                })
            
            if len(numeric_cols) >= 1:
                recommendations.append({
                    "type": "histogram",
                    "reason": "适合展示数值分布",
                    "suggested_x": numeric_cols[0]
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "recommendations": recommendations,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
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
