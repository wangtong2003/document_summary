"""
Prometheus 监控指标模块
提供系统性能指标收集和导出
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics")


class MetricsCollector:
    """指标收集器 - 支持 Prometheus 格式导出"""
    
    def __init__(self):
        self._start_time = time.time()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        # 预定义指标
        self._counters["query_total"] = 0
        self._counters["query_success"] = 0
        self._counters["query_error"] = 0
        self._counters["cache_hits"] = 0
        self._counters["cache_misses"] = 0
        self._counters["mcp_calls"] = 0
        self._counters["llm_requests"] = 0
    
    async def inc_counter(self, name: str, value: int = 1):
        """增加计数器"""
        async with self._lock:
            self._counters[name] += value
    
    async def set_gauge(self, name: str, value: float):
        """设置仪表盘值"""
        async with self._lock:
            self._gauges[name] = value
    
    async def observe_histogram(self, name: str, value: float):
        """记录直方图观测值"""
        async with self._lock:
            self._histograms[name].append(value)
            # 保留最近 1000 个观测值
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
    
    async def record_query(self, success: bool, duration: float):
        """记录查询指标"""
        await self.inc_counter("query_total")
        if success:
            await self.inc_counter("query_success")
        else:
            await self.inc_counter("query_error")
        await self.observe_histogram("query_duration_seconds", duration)
    
    async def record_cache_access(self, hit: bool):
        """记录缓存访问"""
        if hit:
            await self.inc_counter("cache_hits")
        else:
            await self.inc_counter("cache_misses")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        async with self._lock:
            uptime = time.time() - self._start_time
            
            # 计算直方图统计
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    histogram_stats[name] = {
                        "count": n,
                        "sum": sum(values),
                        "mean": sum(values) / n,
                        "p50": sorted_values[int(n * 0.5)] if n > 0 else 0,
                        "p90": sorted_values[int(n * 0.9)] if n > 0 else 0,
                        "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
                        "min": min(values),
                        "max": max(values)
                    }
            
            return {
                "uptime_seconds": uptime,
                "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histogram_stats
            }
    
    def to_prometheus_format(self) -> str:
        """转换为 Prometheus 暴露格式"""
        lines = []
        
        # 帮助信息
        lines.append("# HELP smart_analytics_uptime_seconds Service uptime in seconds")
        lines.append("# TYPE smart_analytics_uptime_seconds gauge")
        lines.append(f"smart_analytics_uptime_seconds {time.time() - self._start_time}")
        
        # 计数器
        counter_help = {
            "query_total": "Total number of queries",
            "query_success": "Total number of successful queries",
            "query_error": "Total number of failed queries",
            "cache_hits": "Total number of cache hits",
            "cache_misses": "Total number of cache misses",
            "mcp_calls": "Total number of MCP calls",
            "llm_requests": "Total number of LLM requests"
        }
        
        for name, value in self._counters.items():
            help_text = counter_help.get(name, f"Metric {name}")
            lines.append(f"# HELP smart_analytics_{name} {help_text}")
            lines.append(f"# TYPE smart_analytics_{name} counter")
            lines.append(f"smart_analytics_{name} {value}")
        
        # 仪表盘
        for name, value in self._gauges.items():
            lines.append(f"# TYPE smart_analytics_{name} gauge")
            lines.append(f"smart_analytics_{name} {value}")
        
        # 直方图
        for name, stats in self._histograms.items():
            if stats:
                lines.append(f"# HELP smart_analytics_{name} {name}")
                lines.append(f"# TYPE smart_analytics_{name} summary")
                lines.append(f'smart_analytics_{name}_count {stats["count"]}')
                lines.append(f'smart_analytics_{name}_sum {stats["sum"]}')
                lines.append(f'smart_analytics_{name}{{quantile="0.5"}} {stats["p50"]}')
                lines.append(f'smart_analytics_{name}{{quantile="0.9"}} {stats["p90"]}')
                lines.append(f'smart_analytics_{name}{{quantile="0.99"}} {stats["p99"]}')
        
        return "\n".join(lines)


# 全局指标收集器实例
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    return metrics_collector
