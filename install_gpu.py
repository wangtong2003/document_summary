#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU支持安装脚本
此脚本帮助用户安装支持GPU的依赖项，优化本地Ollama嵌入
"""

import os
import platform
import subprocess
import sys

def check_gpu():
    """检查是否有可用的GPU"""
    try:
        import torch
        return torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        print("PyTorch未安装，将尝试安装...")
        return False, None
    except Exception as e:
        print(f"检查GPU时出错: {str(e)}")
        return False, None

def install_package(package):
    """安装Python包"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("="*50)
    print("文档摘要系统 - GPU支持安装脚本")
    print("="*50)
    
    # 检查操作系统
    os_name = platform.system()
    print(f"操作系统: {os_name}")
    
    # 先安装基本PyTorch
    try:
        print("\n正在安装PyTorch基础版本...")
        install_package("torch")
    except Exception as e:
        print(f"安装PyTorch失败: {str(e)}")
        return
    
    # 检查GPU可用性
    has_gpu, gpu_name = check_gpu()
    
    if has_gpu:
        print(f"\n检测到GPU: {gpu_name}")
        print("开始安装GPU依赖...")
        
        # 安装PyTorch GPU版本
        print("\n正在安装PyTorch GPU版本...")
        try:
            # 获取CUDA版本
            import torch
            cuda_version = torch.version.cuda
            print(f"CUDA版本: {cuda_version}")
            
            if cuda_version.startswith("11"):
                # CUDA 11.x
                install_package("torch --index-url https://download.pytorch.org/whl/cu118")
            elif cuda_version.startswith("12"):
                # CUDA 12.x
                install_package("torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print(f"未能匹配CUDA版本 {cuda_version}，使用默认安装...")
                install_package("torch")
        except Exception as e:
            print(f"安装PyTorch GPU版本失败: {str(e)}")
        
        # 安装faiss-gpu
        print("\n正在安装faiss-gpu...")
        try:
            install_package("faiss-gpu==1.7.4")
        except Exception as e:
            print(f"安装faiss-gpu失败: {str(e)}")
            print("尝试降级安装faiss-gpu...")
            try:
                install_package("faiss-gpu==1.7.3")
            except Exception as e2:
                print(f"降级安装faiss-gpu失败: {str(e2)}")
    else:
        print("\n未检测到可用的GPU")
        print("不进行GPU依赖安装，将继续使用CPU版本")
    
    # 安装其他必要的依赖
    print("\n正在安装必要的依赖...")
    packages = [
        "langchain-ollama>=0.2.3",
        "langchain-chroma>=0.1.0",
        "chromadb>=0.4.22"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            install_package(package)
        except Exception as e:
            print(f"安装 {package} 失败: {str(e)}")
    
    print("\n"+"="*50)
    print("安装过程完成")
    if has_gpu:
        print(f"GPU支持已配置，使用: {gpu_name}")
        print("\n运行以下命令以验证GPU是否正常工作:")
        print("python -c \"import torch; print('CUDA可用:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')\"")
        print("\n确保Ollama服务已启用并加载了snowflake-arctic-embed2模型:")
        print("ollama pull snowflake-arctic-embed2")
    else:
        print("未配置GPU支持，将使用CPU模式")
    print("="*50)

if __name__ == "__main__":
    main() 