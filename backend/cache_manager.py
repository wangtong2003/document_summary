"""
Redis 缓存管理器 - 带分层缓存优化
用于缓存查询结果和 Schema 信息，提高重复查询性能
支持内存缓存 (L1) + Redis 缓存 (L2) 两层架构
"""

import json
import logging
from typing import Any, Optional, Dict
import hashlib
import asyncio
from datetime import timedelta, datetime
from functools import lru_cache
import time

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache-manager")


class LayeredCacheManager:
    """分层缓存管理器：L1(内存) + L2(Redis)"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 db: int = 0, default_ttl: int = 3600, 
                 l1_max_size: int = 128, l1_ttl: int = 300):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.db = db
        self.default_ttl = default_ttl  # L2 默认缓存时间 (秒)
        self.l1_ttl = l1_ttl            # L1 缓存时间 (秒)
        self.l1_max_size = l1_max_size  # L1 最大条目数
        
        self.redis: Optional[aioredis.Redis] = None
        self._initialized = False
        
        # L1 缓存：使用字典 + 时间戳
        self._l1_cache: Dict[str, Dict[str, Any]] = {}
        self._l1_timestamps: Dict[str, float] = {}
        
        # 统计信息
        self._stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "writes": 0
        }
    
    async def initialize(self) -> bool:
        """初始化 Redis 连接"""
        try:
            self.redis = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{self.db}",
                encoding="utf-8",
                decode_responses=True
            )
            
            # 测试连接
            await self.redis.ping()
            
            self._initialized = True
            logger.info(f"Redis 缓存初始化成功 ({self.redis_host}:{self.redis_port})")
            return True
            
        except Exception as e:
            logger.warning(f"Redis 缓存初始化失败：{e}，将仅使用内存缓存")
            return False
    
    async def close(self):
        """关闭 Redis 连接"""
        if self.redis:
            await self.redis.close()
            self._initialized = False
            logger.info("Redis 缓存已关闭")
    
    def _generate_key(self, query: str, params: Dict = None, prefix: str = "query") -> str:
        """生成缓存键"""
        key_data = f"{prefix}:{query}:{json.dumps(params or {}, sort_keys=True)}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"smart_analytics:{prefix}:{key_hash}"
    
    def _clean_l1_cache(self):
        """清理过期的 L1 缓存"""
        current_time = time.time()
        expired_keys = [
            k for k, ts in self._l1_timestamps.items()
            if current_time - ts > self.l1_ttl
        ]
        for key in expired_keys:
            self._l1_cache.pop(key, None)
            self._l1_timestamps.pop(key, None)
        
        # 如果仍然超出大小限制，移除最旧的条目
        if len(self._l1_cache) > self.l1_max_size:
            sorted_keys = sorted(self._l1_timestamps.keys(), key=lambda k: self._l1_timestamps[k])
            for key in sorted_keys[:len(self._l1_cache) - self.l1_max_size]:
                self._l1_cache.pop(key, None)
                self._l1_timestamps.pop(key, None)
    
    async def get(self, query: str, params: Dict = None, prefix: str = "query") -> Optional[Dict[str, Any]]:
        """从缓存获取数据 (L1 -> L2)"""
        key = self._generate_key(query, params, prefix)
        
        # 清理过期 L1 缓存
        self._clean_l1_cache()
        
        # 尝试 L1 缓存
        if key in self._l1_cache:
            cache_entry = self._l1_cache[key]
            if time.time() - self._l1_timestamps.get(key, 0) < self.l1_ttl:
                self._stats["l1_hits"] += 1
                logger.debug(f"L1 缓存命中：{key}")
                return cache_entry['data']
            else:
                # 过期，移除
                self._l1_cache.pop(key, None)
                self._l1_timestamps.pop(key, None)
        
        self._stats["l1_misses"] += 1
        
        # 尝试 L2 缓存 (Redis)
        if self._initialized and self.redis:
            try:
                cached_data = await self.redis.get(key)
                if cached_data:
                    self._stats["l2_hits"] += 1
                    logger.debug(f"L2 缓存命中：{key}")
                    
                    # 回填到 L1
                    result = json.loads(cached_data)
                    self._l1_cache[key] = {'data': result}
                    self._l1_timestamps[key] = time.time()
                    
                    return result
                
                self._stats["l2_misses"] += 1
            except Exception as e:
                logger.error(f"L2 缓存读取失败：{e}")
        
        logger.debug(f"缓存未命中：{key}")
        return None
    
    async def set(self, query: str, result: Dict[str, Any], 
                  params: Dict = None, ttl: int = None, prefix: str = "query") -> bool:
        """缓存数据到 L1 和 L2"""
        key = self._generate_key(query, params, prefix)
        ttl = ttl or self.default_ttl
        
        # 写入 L1 缓存
        self._clean_l1_cache()
        self._l1_cache[key] = {'data': result}
        self._l1_timestamps[key] = time.time()
        
        # 写入 L2 缓存 (Redis)
        if self._initialized and self.redis:
            try:
                await self.redis.setex(
                    key,
                    ttl,
                    json.dumps(result, ensure_ascii=False, default=str)
                )
                logger.debug(f"缓存已设置：{key} (TTL={ttl}s)")
            except Exception as e:
                logger.error(f"L2 缓存写入失败：{e}")
                return False
        
        self._stats["writes"] += 1
        return True
    
    async def delete(self, query: str, params: Dict = None, prefix: str = "query") -> bool:
        """删除缓存"""
        key = self._generate_key(query, params, prefix)
        
        # 删除 L1 缓存
        self._l1_cache.pop(key, None)
        self._l1_timestamps.pop(key, None)
        
        # 删除 L2 缓存
        if self._initialized and self.redis:
            try:
                await self.redis.delete(key)
                logger.debug(f"缓存已删除：{key}")
                return True
            except Exception as e:
                logger.error(f"L2 缓存删除失败：{e}")
                return False
        
        return True
    
    async def clear_all(self) -> bool:
        """清空所有缓存"""
        # 清空 L1
        self._l1_cache.clear()
        self._l1_timestamps.clear()
        
        # 清空 L2
        if self._initialized and self.redis:
            try:
                pattern = "smart_analytics:*"
                keys = []
                async for key in self.redis.scan_iter(match=pattern):
                    keys.append(key)
                
                if keys:
                    await self.redis.delete(*keys)
                    logger.info(f"已清空 {len(keys)} 条 L2 缓存记录")
                
                return True
            except Exception as e:
                logger.error(f"清空 L2 缓存失败：{e}")
                return False
        
        logger.info("已清空 L1 缓存")
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_reads = self._stats["l1_hits"] + self._stats["l1_misses"]
        l1_hit_rate = (self._stats["l1_hits"] / total_reads * 100) if total_reads > 0 else 0
        
        total_l2_reads = self._stats["l2_hits"] + self._stats["l2_misses"]
        l2_hit_rate = (self._stats["l2_hits"] / total_l2_reads * 100) if total_l2_reads > 0 else 0
        
        stats = {
            "enabled": self._initialized,
            "l1_size": len(self._l1_cache),
            "l1_max_size": self.l1_max_size,
            "l1_hits": self._stats["l1_hits"],
            "l1_misses": self._stats["l1_misses"],
            "l1_hit_rate": round(l1_hit_rate, 2),
            "l2_hits": self._stats["l2_hits"],
            "l2_misses": self._stats["l2_misses"],
            "l2_hit_rate": round(l2_hit_rate, 2),
            "total_writes": self._stats["writes"]
        }
        
        # 添加 Redis 统计 (如果可用)
        if self._initialized and self.redis:
            try:
                info = await self.redis.info("stats")
                keys_count = await self.redis.dbsize()
                stats["l2_keys_count"] = keys_count
                stats["l2_redis_hits"] = info.get("keyspace_hits", 0)
                stats["l2_redis_misses"] = info.get("keyspace_misses", 0)
            except Exception as e:
                stats["l2_error"] = str(e)
        
        return stats


# Schema 专用缓存 (长时间缓存)
class SchemaCacheManager:
    """Schema 缓存管理器 - 专门用于缓存数据库表结构"""
    
    def __init__(self, base_cache: LayeredCacheManager):
        self.base_cache = base_cache
        self.schema_ttl = 7200  # Schema 缓存 2 小时
    
    async def get_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """获取表结构缓存"""
        return await self.base_cache.get(table_name, prefix="schema")
    
    async def set_schema(self, table_name: str, schema: Dict[str, Any]) -> bool:
        """缓存表结构"""
        return await self.base_cache.set(table_name, schema, ttl=self.schema_ttl, prefix="schema")
    
    async def clear_schemas(self) -> bool:
        """清空所有 Schema 缓存"""
        return await self.base_cache.clear_all()


# 全局缓存实例
cache_manager = LayeredCacheManager()
schema_cache = SchemaCacheManager(cache_manager)


async def init_cache(redis_host: str = 'localhost', redis_port: int = 6379, 
                     **kwargs) -> bool:
    """初始化全局缓存管理器"""
    global cache_manager, schema_cache
    cache_manager = LayeredCacheManager(redis_host, redis_port, **kwargs)
    schema_cache = SchemaCacheManager(cache_manager)
    return await cache_manager.initialize()


def get_cache_manager() -> LayeredCacheManager:
    """获取全局缓存管理器实例"""
    return cache_manager


def get_schema_cache() -> SchemaCacheManager:
    """获取 Schema 缓存管理器实例"""
    return schema_cache
