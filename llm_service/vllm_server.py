#!/usr/bin/env python3
"""
vLLM 服务启动脚本
支持直接运行或作为 Docker 入口点
"""

import os
import sys
import yaml
import argparse
from typing import Optional

def load_config(config_path: str = "config/vllm_config.yaml") -> dict:
    """加载 vLLM 配置文件"""
    if not os.path.exists(config_path):
        print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_vllm_command(config: dict) -> list:
    """根据配置构建 vLLM 启动命令"""
    model_config = config.get('model', {})
    server_config = config.get('server', {})
    perf_config = config.get('performance', {})
    guided_config = config.get('guided_decoding', {})
    api_config = config.get('api', {})
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_config.get('name', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
        "--host", server_config.get('host', '0.0.0.0'),
        "--port", str(server_config.get('port', 8000)),
        "--tensor-parallel-size", str(perf_config.get('tensor_parallel_size', 1)),
        "--max-model-len", str(perf_config.get('max_model_len', 8192)),
        "--gpu-memory-utilization", str(perf_config.get('gpu_memory_utilization', 0.9)),
        "--max-num-seqs", str(perf_config.get('max_num_seqs', 256)),
    ]
    
    # 性能优化选项
    if perf_config.get('enable_chunked_prefill', False):
        cmd.append("--enable-chunked-prefill")
    
    if perf_config.get('num_scheduler_steps', 1) > 1:
        cmd.extend(["--num-scheduler-steps", str(perf_config['num_scheduler_steps'])])
    
    # 引导解码配置
    if guided_config.get('enabled', False):
        cmd.extend([
            "--guided-decoding-backend",
            guided_config.get('backend', 'outlines')
        ])
    
    # API 配置
    if api_config.get('api_key'):
        cmd.extend(["--api-key", api_config['api_key']])
    
    if api_config.get('timeout'):
        cmd.extend(["--request-timeout", str(api_config['timeout'])])
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description='vLLM 服务启动器')
    parser.add_argument('--config', type=str, default='config/vllm_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                       help='覆盖模型名称')
    parser.add_argument('--port', type=int, default=None,
                       help='覆盖服务端口')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.model:
        config.setdefault('model', {})['name'] = args.model
    if args.port:
        config.setdefault('server', {})['port'] = args.port
    
    # 构建并执行命令
    cmd = build_vllm_command(config)
    print(f"启动 vLLM 服务：{' '.join(cmd)}")
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"vLLM 服务启动失败：{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()
