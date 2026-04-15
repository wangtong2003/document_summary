"""
LangGraph 智能问数 Agent 图构建
"""

import logging
from typing import Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .state import AgentState
from .nodes import AgentNodes, create_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph-graph")


def build_agent_graph(vllm_api_key: str = "vllm_api_key", 
                      vllm_base_url: str = "http://localhost:8000/v1",
                      model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct") -> StateGraph:
    """
    构建 LangGraph 智能问数 Agent
    
    Args:
        vllm_api_key: vLLM API 密钥
        vllm_base_url: vLLM 服务地址
        model_name: 模型名称
    
    Returns:
        编译好的 StateGraph
    """
    
    # 初始化 LLM 客户端 (使用 vLLM OpenAI 兼容接口)
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=vllm_api_key,
        openai_api_base=vllm_base_url,
        temperature=0.1,  # 低温度保证 SQL 生成稳定性
        max_tokens=2048
    )
    
    # 创建节点实例
    nodes = AgentNodes(llm)
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("parse_query", nodes.parse_query)
    workflow.add_node("get_db_schema", nodes.get_db_schema)
    workflow.add_node("generate_sql", nodes.generate_sql)
    workflow.add_node("validate_sql", nodes.validate_sql)
    workflow.add_node("execute_query", nodes.execute_query)
    workflow.add_node("analyze_data", nodes.analyze_data)
    workflow.add_node("recommend_charts", nodes.recommend_charts)
    workflow.add_node("create_chart", nodes.create_chart)
    workflow.add_node("generate_response", nodes.generate_response)
    workflow.add_node("handle_error", nodes.handle_error)
    
    # 设置入口点
    workflow.set_entry_point("parse_query")
    
    # 添加条件边 (使用路由函数)
    workflow.add_conditional_edges(
        "parse_query",
        create_router,
        {
            "get_db_schema": "get_db_schema",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "get_db_schema",
        create_router,
        {
            "generate_sql": "generate_sql",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_sql",
        create_router,
        {
            "validate_sql": "validate_sql",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "validate_sql",
        create_router,
        {
            "execute_query": "execute_query",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_query",
        create_router,
        {
            "analyze_data": "analyze_data",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_data",
        create_router,
        {
            "recommend_charts": "recommend_charts",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "recommend_charts",
        create_router,
        {
            "create_chart": "create_chart",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "create_chart",
        create_router,
        {
            "generate_response": "generate_response",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        create_router,
        {
            "__end__": END,
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "handle_error",
        create_router,
        {
            "sql_generation": "generate_sql",
            "execute_query": "execute_query",
            "__end__": END
        }
    )
    
    # 编译图
    app = workflow.compile()
    
    logger.info("LangGraph Agent 构建完成")
    return app


async def run_agent(query: str, conversation_history: list = None) -> dict:
    """
    运行智能问数 Agent
    
    Args:
        query: 用户查询
        conversation_history: 对话历史
    
    Returns:
        Agent 执行结果
    """
    
    # 构建图
    app = build_agent_graph()
    
    # 初始化状态
    initial_state = {
        "user_query": query,
        "conversation_history": conversation_history or [],
        "db_schema": None,
        "generated_sql": None,
        "sql_validation_result": None,
        "query_result": None,
        "query_error": None,
        "data_summary": None,
        "analysis_notes": [],
        "chart_recommendations": [],
        "selected_chart_type": None,
        "chart_config": None,
        "chart_html": None,
        "final_response": None,
        "current_step": "start",
        "retry_count": 0,
        "needs_correction": False
    }
    
    # 运行 Agent
    result = await app.ainvoke(initial_state)
    
    return result
