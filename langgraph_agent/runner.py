"""
LangGraph Agent 图构建和运行入口
集成 MCP 客户端、缓存和连接池
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import AgentNodes
from backend.mcp_client import mcp_client, get_mcp_client
from backend.cache_manager import get_cache_manager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-graph")


class SmartAnalyticsAgent:
    """智能问数 Agent 管理器"""
    
    def __init__(self):
        self.llm: Optional[ChatOpenAI] = None
        self.graph = None
        self._initialized = False
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 客户端（带超时控制）"""
        return ChatOpenAI(
            model=settings.VLLM_MODEL_NAME,
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY,
            temperature=0.1,  # 降低温度提高 SQL 准确性
            request_timeout=60,  # P1: LLM 超时控制
            max_retries=2
        )
    
    async def initialize(self) -> bool:
        """初始化 Agent"""
        try:
            logger.info("初始化智能问数 Agent...")
            
            # 初始化 LLM
            self.llm = self._create_llm()
            
            # 初始化 MCP 客户端
            mcp_initialized = await mcp_client.initialize()
            if not mcp_initialized:
                logger.error("MCP 客户端初始化失败")
                return False
            
            # 初始化缓存 (可选)
            cache = get_cache_manager()
            await cache.initialize()
            
            # 构建 LangGraph
            self._build_graph()
            
            self._initialized = True
            logger.info("智能问数 Agent 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"Agent 初始化失败：{e}")
            return False
    
    def _build_graph(self):
        """构建 LangGraph 状态机"""
        nodes = AgentNodes(self.llm)
        
        # 创建工作流图
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
        
        # 设置入口
        workflow.set_entry_point("parse_query")
        
        # 添加边 (使用条件路由)
        workflow.add_conditional_edges(
            "parse_query",
            lambda state: "get_db_schema",
            ["get_db_schema"]
        )
        
        workflow.add_conditional_edges(
            "get_db_schema",
            lambda state: "generate_sql",
            ["generate_sql"]
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            lambda state: "validate_sql",
            ["validate_sql"]
        )
        
        workflow.add_conditional_edges(
            "validate_sql",
            lambda state: "execute_query" if state.get('sql_validation_result') else "handle_error",
            ["execute_query", "handle_error"]
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            lambda state: "analyze_data" if state.get('query_result') else "handle_error",
            ["analyze_data", "handle_error"]
        )
        
        workflow.add_edge("analyze_data", "recommend_charts")
        workflow.add_edge("recommend_charts", "create_chart")
        workflow.add_edge("create_chart", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # 错误处理路由
        workflow.add_conditional_edges(
            "handle_error",
            lambda state: state.get('current_step'),
            ["sql_generation", "execute_query", "generate_response", "__end__"]
        )
        
        # 编译图
        self.graph = workflow.compile()
        logger.info("LangGraph 工作流构建完成")
    
    async def run(self, user_query: str) -> Dict[str, Any]:
        """运行 Agent 处理用户查询"""
        if not self._initialized:
            return {
                "success": False,
                "error": "Agent 未初始化"
            }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # P1: 检查缓存
            cache = get_cache_manager()
            cached_result = await cache.get(user_query)
            if cached_result:
                logger.info(f"缓存命中，直接返回结果")
                cached_result['from_cache'] = True
                return cached_result
            
            # 初始化状态
            initial_state: AgentState = {
                "user_query": user_query,
                "current_step": "start",
                "retry_count": 0,
                "needs_correction": False
            }
            
            # 运行工作流
            logger.info(f"开始处理查询：{user_query}")
            result = await self.graph.ainvoke(initial_state)
            
            # 提取关键信息
            response = {
                "success": result.get('current_step') == 'complete',
                "final_response": result.get('final_response', ''),
                "generated_sql": result.get('generated_sql', ''),
                "query_result": result.get('query_result', []),
                "data_summary": result.get('data_summary', {}),
                "chart_html": result.get('chart_html'),
                "chart_config": result.get('chart_config'),
                "analysis_notes": result.get('analysis_notes', []),
                "query_error": result.get('query_error'),
                "execution_time": asyncio.get_event_loop().time() - start_time
            }
            
            # P1: 缓存成功结果
            if response["success"] and response["query_result"]:
                await cache.set(user_query, response, ttl=3600)
            
            return response
            
        except Exception as e:
            logger.error(f"Agent 执行失败：{e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time": asyncio.get_event_loop().time() - start_time
            }
    
    async def shutdown(self):
        """关闭 Agent 和资源"""
        logger.info("关闭智能问数 Agent...")
        
        # 关闭 MCP 客户端
        await mcp_client.shutdown()
        
        # 关闭缓存
        cache = get_cache_manager()
        await cache.close()
        
        self._initialized = False
        logger.info("智能问数 Agent 已关闭")


# 全局 Agent 实例
agent = SmartAnalyticsAgent()


async def init_agent() -> bool:
    """初始化全局 Agent"""
    return await agent.initialize()


async def run_agent(user_query: str) -> Dict[str, Any]:
    """运行 Agent 处理查询"""
    return await agent.run(user_query)


async def shutdown_agent():
    """关闭全局 Agent"""
    await agent.shutdown()
