"""
LangGraph 智能问数 Agent 节点实现
集成真实 MCP 客户端调用
"""

import json
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .state import AgentState
from backend.mcp_client import mcp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph-nodes")

# 系统提示词模板
SQL_GENERATION_PROMPT = """你是一个专业的数据分析师和 SQL 专家。请根据用户问题和数据库结构生成准确的 SQL 查询语句。

数据库表结构：
{db_schema}

要求：
1. 只生成 SELECT 语句，不允许修改数据
2. 使用适当的 JOIN 连接相关表
3. 添加合理的 WHERE 条件过滤
4. 包含必要的聚合函数 (COUNT, SUM, AVG 等)
5. 添加 ORDER BY 和 LIMIT 限制结果数量
6. 输出纯 SQL 语句，不要包含解释
7. 如果表名或列名包含特殊字符，使用反引号包裹

用户问题：{user_query}

请生成 SQL 查询："""

DATA_ANALYSIS_PROMPT = """你是一个数据分析专家。请分析以下查询结果并提供洞察。

查询结果：
{query_result}

请提供：
1. 数据概览（行数、主要字段）
2. 关键发现
3. 趋势或模式
4. 建议的可视化方式"""

CHART_SELECTION_PROMPT = """你是一个数据可视化专家。根据以下数据和分析，推荐最合适的图表类型。

数据摘要：
{data_summary}

分析笔记：
{analysis_notes}

可用图表类型：bar, line, scatter, pie, area, histogram, box, heatmap, treemap, funnel

请推荐 1-3 种图表类型，并说明理由。输出 JSON 格式：
{{
    "recommended_charts": [
        {{"type": "chart_type", "reason": "理由"}}
    ],
    "best_choice": "最佳图表类型"
}}"""


class AgentNodes:
    """LangGraph Agent 节点集合"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm = llm_client
    
    async def parse_query(self, state: AgentState) -> AgentState:
        """解析用户查询，提取意图"""
        logger.info(f"解析查询：{state['user_query']}")
        
        # 这里可以添加更复杂的意图识别逻辑
        # 目前简单标记为需要查询数据库
        state['current_step'] = 'schema_retrieval'
        state['retry_count'] = 0
        
        return state
    
    async def get_db_schema(self, state: AgentState) -> AgentState:
        """获取数据库表结构（通过 MCP 协议）- 并行化 + Schema 缓存优化"""
        logger.info("获取数据库表结构")
        
        try:
            # 通过 MCP 客户端获取所有表
            tables_result = await mcp_client.list_tables()
            
            if not tables_result.get('success'):
                logger.warning(f"获取表列表失败：{tables_result.get('error')}")
                state['db_schema'] = {"tables": [], "schema_info": "无法获取表结构"}
                state['current_step'] = 'sql_generation'
                return state
            
            tables = tables_result.get('tables', [])
            
            # 从 Schema 缓存获取 (优化点：长时间缓存表结构)
            from backend.cache_manager import get_schema_cache
            schema_cache = get_schema_cache()
            
            # 并行获取每个表的详细结构，优先从缓存读取
            async def get_schema_with_cache(table_name: str):
                # 先尝试缓存
                cached = await schema_cache.get_schema(table_name)
                if cached:
                    logger.debug(f"Schema 缓存命中：{table_name}")
                    return {"success": True, "schema": cached, "from_cache": True}
                
                # 缓存未命中，从 MCP 获取
                result = await mcp_client.get_table_schema(table_name)
                if result.get('success'):
                    # 写入缓存 (2 小时 TTL)
                    await schema_cache.set_schema(table_name, result.get('schema', []))
                    logger.debug(f"Schema 已缓存：{table_name}")
                return result
            
            # 并行执行所有表的 schema 获取
            schema_tasks = [get_schema_with_cache(table) for table in tables]
            schema_results = await asyncio.gather(*schema_tasks, return_exceptions=True)
            
            # 处理结果
            schema_details = {}
            cache_hits = 0
            for table, result in zip(tables, schema_results):
                if isinstance(result, Exception):
                    logger.error(f"获取表 {table} 结构失败：{result}")
                    continue
                if result.get('success'):
                    schema_details[table] = result.get('schema', [])
                    if result.get('from_cache'):
                        cache_hits += 1
                else:
                    logger.warning(f"获取表 {table} 结构失败：{result.get('error')}")
            
            logger.info(f"Schema 获取完成：{len(tables)}个表，缓存命中 {cache_hits} 个")
            
            state['db_schema'] = {
                "tables": tables,
                "schema_info": schema_details
            }
            state['current_step'] = 'sql_generation'
            
        except Exception as e:
            logger.error(f"获取数据库结构失败：{e}")
            state['db_schema'] = {"tables": [], "schema_info": f"获取失败：{str(e)}"}
            state['current_step'] = 'sql_generation'
        
        return state
    
    async def generate_sql(self, state: AgentState) -> AgentState:
        """生成 SQL 查询语句"""
        logger.info("生成 SQL 查询")
        
        # 格式化 schema 信息
        db_schema = state.get('db_schema', {})
        schema_text = ""
        for table_name, columns in db_schema.get('schema_info', {}).items():
            schema_text += f"表名：{table_name}\n"
            for col in columns:
                schema_text += f"  - {col.get('Field', 'N/A')}: {col.get('Type', 'N/A')} ({col.get('Key', '')})\n"
            schema_text += "\n"
        
        prompt = SQL_GENERATION_PROMPT.format(
            db_schema=schema_text or "无表结构信息",
            user_query=state['user_query']
        )
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            sql_query = response.content.strip()
            
            # 清理 SQL 语句，移除 markdown 代码块标记
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            state['generated_sql'] = sql_query
            state['current_step'] = 'sql_validation'
            
        except Exception as e:
            logger.error(f"SQL 生成失败：{e}")
            state['query_error'] = f"SQL 生成失败：{str(e)}"
            state['needs_correction'] = True
            state['current_step'] = 'error_handling'
        
        return state
    
    async def validate_sql(self, state: AgentState) -> AgentState:
        """验证 SQL 语句安全性"""
        logger.info("验证 SQL 语句")
        
        sql = state.get('generated_sql', '').upper()
        
        # 安全检查
        if not sql.startswith('SELECT'):
            state['sql_validation_result'] = False
            state['query_error'] = "只允许 SELECT 查询"
            state['needs_correction'] = True
            state['current_step'] = 'error_handling'
            return state
        
        # 禁止危险操作
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql:
                state['sql_validation_result'] = False
                state['query_error'] = f"检测到危险操作：{keyword}"
                state['needs_correction'] = True
                state['current_step'] = 'error_handling'
                return state
        
        state['sql_validation_result'] = True
        state['current_step'] = 'execute_query'
        
        return state
    
    async def execute_query(self, state: AgentState) -> AgentState:
        """执行数据库查询（通过 MCP 协议）"""
        logger.info("执行数据库查询")
        
        try:
            # 通过 MCP 客户端执行 SQL 查询
            sql_query = state.get('generated_sql', '')
            result = await mcp_client.execute_sql_query(sql_query, limit=100)
            
            if result.get('success'):
                state['query_result'] = result.get('data', [])
                state['query_row_count'] = result.get('row_count', 0)
                state['current_step'] = 'data_analysis'
            else:
                state['query_error'] = result.get('error', '查询失败')
                state['needs_correction'] = True
                state['current_step'] = 'error_handling'
                
        except Exception as e:
            logger.error(f"查询执行失败：{e}")
            state['query_error'] = str(e)
            state['needs_correction'] = True
            state['current_step'] = 'error_handling'
        
        return state
    
    async def analyze_data(self, state: AgentState) -> AgentState:
        """分析查询结果"""
        logger.info("分析数据")
        
        query_result = state.get('query_result', [])
        
        if not query_result:
            state['analysis_notes'] = ["无数据可分析"]
            state['current_step'] = 'chart_recommendation'
            return state
        
        # 数据统计
        row_count = len(query_result)
        columns = list(query_result[0].keys()) if query_result else []
        
        state['data_summary'] = {
            "row_count": row_count,
            "columns": columns,
            "sample": query_result[:5] if row_count > 5 else query_result
        }
        
        # 使用 LLM 进行分析
        prompt = DATA_ANALYSIS_PROMPT.format(
            query_result=json.dumps(query_result, ensure_ascii=False, default=str)
        )
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            state['analysis_notes'] = [response.content]
        except Exception as e:
            logger.error(f"数据分析失败：{e}")
            state['analysis_notes'] = ["数据分析失败"]
        
        state['current_step'] = 'chart_recommendation'
        return state
    
    async def recommend_charts(self, state: AgentState) -> AgentState:
        """推荐图表类型"""
        logger.info("推荐图表")
        
        prompt = CHART_SELECTION_PROMPT.format(
            data_summary=json.dumps(state.get('data_summary', {}), ensure_ascii=False),
            analysis_notes="\n".join(state.get('analysis_notes', []))
        )
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            recommendations = json.loads(response.content)
            
            state['chart_recommendations'] = recommendations.get('recommended_charts', [])
            state['selected_chart_type'] = recommendations.get('best_choice', 'bar')
            state['current_step'] = 'create_chart'
            
        except Exception as e:
            logger.error(f"图表推荐失败：{e}")
            state['chart_recommendations'] = []
            state['selected_chart_type'] = 'bar'  # 默认柱状图
            state['current_step'] = 'create_chart'
        
        return state
    
    async def create_chart(self, state: AgentState) -> AgentState:
        """创建可视化图表（通过 MCP 协议）"""
        logger.info(f"创建图表：{state['selected_chart_type']}")
        
        try:
            # 通过 MCP 客户端调用可视化工具
            query_result = state.get('query_result', [])
            chart_type = state.get('selected_chart_type', 'bar')
            data_summary = state.get('data_summary', {})
            columns = data_summary.get('columns', [])
            
            # 自动推断 X/Y 轴
            x_column = columns[0] if len(columns) > 0 else None
            y_column = columns[1] if len(columns) > 1 else columns[0]
            
            result = await mcp_client.create_chart(
                chart_type=chart_type,
                data=query_result,
                x_column=x_column,
                y_column=y_column,
                title=f"{state['user_query']} - 可视化",
                width=800,
                height=600
            )
            
            if result.get('success'):
                state['chart_html'] = result.get('chart_html', None)
                state['chart_config'] = {
                    "type": chart_type,
                    "data_columns": columns,
                    "title": f"{state['user_query']} - 可视化"
                }
            else:
                logger.warning(f"图表创建失败：{result.get('error')}")
                state['chart_html'] = None
                state['chart_config'] = None
            
            state['current_step'] = 'generate_response'
            
        except Exception as e:
            logger.error(f"图表创建失败：{e}")
            state['chart_html'] = None
            state['chart_config'] = None
            state['current_step'] = 'generate_response'
        
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终响应"""
        logger.info("生成最终响应")
        
        response_parts = []
        
        # 添加查询结果摘要
        if state.get('query_result'):
            row_count = len(state['query_result'])
            response_parts.append(f"查询到 {row_count} 条记录。")
        
        # 添加分析结果
        if state.get('analysis_notes'):
            response_parts.append("\n**数据分析**:")
            response_parts.extend(state['analysis_notes'])
        
        # 添加图表信息
        if state.get('chart_html'):
            response_parts.append(f"\n**可视化**: 已生成{state['selected_chart_type']}图表")
        
        state['final_response'] = "\n".join(response_parts)
        state['current_step'] = 'complete'
        
        return state
    
    async def handle_error(self, state: AgentState) -> AgentState:
        """错误处理"""
        logger.warning(f"错误处理：{state.get('query_error')}")
        
        retry_count = state.get('retry_count', 0)
        
        if retry_count < 3:
            # 尝试重试
            state['retry_count'] = retry_count + 1
            state['needs_correction'] = False
            
            # 根据错误类型决定回到哪一步
            if state.get('query_error') and 'SQL' in state['query_error']:
                state['current_step'] = 'sql_generation'
            else:
                state['current_step'] = 'execute_query'
        else:
            # 超过最大重试次数，返回错误信息
            state['final_response'] = f"抱歉，处理您的请求时遇到错误：{state.get('query_error', '未知错误')}"
            state['current_step'] = 'complete'
        
        return state


def create_router(state: AgentState) -> str:
    """路由决策函数"""
    current_step = state.get('current_step', 'start')
    
    # 定义状态流转
    step_mapping = {
        'start': 'parse_query',
        'parse_query': 'get_db_schema',
        'schema_retrieval': 'get_db_schema',
        'get_db_schema': 'generate_sql',
        'sql_generation': 'generate_sql',
        'generate_sql': 'validate_sql',
        'sql_validation': 'validate_sql',
        'validate_sql': 'execute_query',
        'execute_query': 'execute_query',
        'data_analysis': 'analyze_data',
        'analyze_data': 'recommend_charts',
        'chart_recommendation': 'recommend_charts',
        'recommend_charts': 'create_chart',
        'create_chart': 'create_chart',
        'generate_response': 'generate_response',
        'error_handling': 'handle_error',
        'complete': '__end__'
    }
    
    next_step = step_mapping.get(current_step, '__end__')
    
    # 检查是否需要修正
    if state.get('needs_correction', False):
        return 'handle_error'
    
    return next_step
