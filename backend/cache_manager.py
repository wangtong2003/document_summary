"""
Redis 缓存管理模块
用于缓存查询结果，提高重复查询性能
"""

import json
import logging
from typing import Any, Optional, Dict
import hashlib
import asyncio
from datetime import timedelta

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache-manager")


class QueryCacheManager:
    """查询缓存管理器"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 db: int = 0, default_ttl: int = 3600):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.db = db
        self.default_ttl = default_ttl  # 默认缓存时间 (秒)
        self.redis: Optional[aioredis.Redis] = None
        self._initialized = False
    
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
            logger.warning(f"Redis 缓存初始化失败：{e}，将不使用缓存")
            return False
    
    async def close(self):
        """关闭 Redis 连接"""
        if self.redis:
            await self.redis.close()
            self._initialized = False
            logger.info("Redis 缓存已关闭")
    
    def _generate_key(self, query: str, params: Dict = None) -> str:
        """生成缓存键"""
        key_data = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"smart_analytics:query:{key_hash}"
    
    async def get(self, query: str, params: Dict = None) -> Optional[Dict[str, Any]]:
        """从缓存获取查询结果"""
        if not self._initialized:
            return None
        
        try:
            key = self._generate_key(query, params)
            cached_data = await self.redis.get(key)
            
            if cached_data:
                logger.debug(f"缓存命中：{key}")
                return json.loads(cached_data)
            
            logger.debug(f"缓存未命中：{key}")
            return None
            
        except Exception as e:
            logger.error(f"缓存读取失败：{e}")
            return None
    
    async def set(self, query: str, result: Dict[str, Any], 
                  params: Dict = None, ttl: int = None) -> bool:
        """缓存查询结果"""
        if not self._initialized:
            return False
        
        try:
            key = self._generate_key(query, params)
            ttl = ttl or self.default_ttl
            
            await self.redis.setex(
                key,
                ttl,
                json.dumps(result, ensure_ascii=False, default=str)
            )
            
            logger.debug(f"缓存已设置：{key} (TTL={ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"缓存写入失败：{e}")
            return False
    
    async def delete(self, query: str, params: Dict = None) -> bool:
        """删除缓存"""
        if not self._initialized:
            return False
        
        try:
            key = self._generate_key(query, params)
            await self.redis.delete(key)
            logger.debug(f"缓存已删除：{key}")
            return True
            
        except Exception as e:
            logger.error(f"缓存删除失败：{e}")
            return False
    
    async def clear_all(self) -> bool:
        """清空所有缓存"""
        if not self._initialized:
            return False
        
        try:
            pattern = "smart_analytics:query:*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"已清空 {len(keys)} 条缓存记录")
            
            return True
            
        except Exception as e:
            logger.error(f"清空缓存失败：{e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self._initialized:
            return {"enabled": False}
        
        try:
            info = await self.redis.info("stats")
            keys_count = await self.redis.dbsize()
            
            return {
                "enabled": True,
                "keys_count": keys_count,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
            
        except Exception as e:
            logger.error(f"获取缓存统计失败：{e}")
            return {"enabled": False, "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """计算缓存命中率"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round(hits / total * 100, 2)


# 全局缓存实例
cache_manager = QueryCacheManager()


async def init_cache(redis_host: str = 'localhost', redis_port: int = 6379, 
                     **kwargs) -> bool:
    """初始化全局缓存管理器"""
    global cache_manager
    cache_manager = QueryCacheManager(redis_host, redis_port, **kwargs)
    return await cache_manager.initialize()


def get_cache_manager() -> QueryCacheManager:
    """获取全局缓存管理器实例"""
    return cache_manager
