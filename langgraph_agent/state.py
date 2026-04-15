"""
LangGraph 智能问数 Agent 状态定义
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    """Agent 状态管理"""
    
    # 用户输入
    user_query: str
    
    # 对话历史
    conversation_history: List[Dict[str, str]]
    
    # 数据库相关
    db_schema: Optional[Dict[str, Any]]  # 数据库表结构
    generated_sql: Optional[str]         # 生成的 SQL 语句
    sql_validation_result: Optional[bool]  # SQL 验证结果
    query_result: Optional[List[Dict]]   # 查询结果
    query_error: Optional[str]           # 查询错误信息
    
    # 数据分析
    data_summary: Optional[Dict[str, Any]]  # 数据统计摘要
    analysis_notes: List[str]              # 分析笔记
    
    # 可视化相关
    chart_recommendations: List[Dict]    # 图表推荐
    selected_chart_type: Optional[str]   # 选定的图表类型
    chart_config: Optional[Dict]         # 图表配置
    chart_html: Optional[str]            # 生成的图表 HTML
    
    # 最终响应
    final_response: Optional[str]
    
    # 流程控制
    current_step: str                    # 当前步骤
    retry_count: int                     # 重试次数
    needs_correction: bool               # 是否需要修正
