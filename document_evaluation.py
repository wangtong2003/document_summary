#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文档摘要参数评估工具
整合所有评估功能于一个文件中
"""

import os
import sys
import json
import time
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from ollama import Client

# 导入现有项目功能
from app import read_document, generate_keywords_with_model, get_default_keywords

# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建API客户端
def create_openai_client():
    """创建并返回OpenAI API客户端"""
    return OpenAI(
        # 使用阿里云通义千问的API密钥
        api_key="sk-c278401b29864808b81c3d7c841300e1",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# 预创建客户端实例
openai_client = create_openai_client()

#--------- 配置部分 ---------#

# 评估参数配置
PARAMETER_CONFIGS = [
    {"temperature": 0.1, "top_p": 0.9, "top_k": 40, "name": "低温度高精确"},
    {"temperature": 0.5, "top_p": 0.8, "top_k": 30, "name": "中温度均衡"},
    {"temperature": 0.8, "top_p": 0.6, "top_k": 20, "name": "高温度创意"},
    {"temperature": 0.3, "top_p": 1.0, "top_k": 50, "name": "低温完整采样"},
    {"temperature": 0.7, "top_p": 0.7, "top_k": 10, "name": "高温缩小范围"}
]

# 测试文档路径
TEST_DOCS_PATH = r"D:\文档摘要测试"

# 结果文件夹路径
RESULTS_FOLDER = "evaluation_results"

# 是否覆盖旧结果（True：总是覆盖，False：每次生成新文件）
OVERWRITE_RESULTS = True

# 默认要评估的文件列表（为空表示评估所有文件）
TARGET_FILES = []

# 评估提示词
EVALUATION_PROMPT = """
你是一个专业文档评估专家，需要评估摘要质量。请通过以下思维链步骤对摘要质量进行客观、一致的评估：

[思维链评估流程]
1. 理解原文：仔细阅读原文内容，识别重要概念、事实、数据和主要观点
2. 理解摘要：阅读生成的摘要，与原文进行对比
3. 信息准确性分析：逐条检查摘要中的事实是否与原文一致
4. 信息召回率计算：计算摘要包含的原文重要信息比例
5. 信息冗余检查：检查摘要是否添加了原文中不存在的内容
6. 信息质量分析：评估可读性和重复性问题
7. 评分汇总：根据客观标准给出最终评分

[评分标准]
1. 信息准确性（1分）：
   - 第一步：列出摘要中包含的关键事实点（至少5个）
   - 第二步：逐一核对这些事实点与原文的一致性
   - 第三步：准确率 = 正确事实点数量/总事实点数量
   - 计分方式：完全准确 = 1分，80%准确 = 0.8分，依此类推
   
2. 信息召回（1分）：
   - 第一步：列出原文中的关键信息点（至少10个）
   - 第二步：检查摘要中包含了多少关键信息点
   - 第三步：计算召回率 = 摘要包含的关键信息点/原文关键信息点总数
   - 计分方式：召回率 ≥ 80% = 1分，60% = 0.6分，依此类推
   
3. 信息冗余（1分）：
   - 第一步：列出摘要中的每个段落或关键点
   - 第二步：逐一检查是否有原文中不存在的内容
   - 第三步：标记所有不属于原文的内容
   - 计分方式：无冗余信息 = 1分，有任何冗余信息 = 0分
   
4. 信息质量（1分）：
   - 第一步：检查标点、符号、错别字等形式问题
   - 第二步：检查是否有重复表达同一信息的情况
   - 第三步：评估摘要的整体结构和连贯性
   - 计分方式：无可读性问题且无重复 = 1分，有问题 = 0分

[评分一致性要求]
- 请始终使用相同的标准进行评分，确保评分过程客观且可重现
- 对于每个评分维度，请详细记录你的思考过程和判断理由
- 在最终计算总分前，请复核各项评分是否符合标准定义

请先按照上述思维链，详细分析每项指标的得分理由，然后给出最终评分。你的回复必须包含一个JSON格式的评分结果：
```json
{{
  "信息准确性": 0.8,
  "信息召回": 0.5,
  "信息冗余": 1,
  "信息质量": 0,
  "总分": 2.3,
  "分析": "在这里给出详细分析..."
}}
```

原文内容：
{original_text}

摘要内容：
{summary_text}
"""

#--------- 功能函数 ---------#

def read_test_documents(docs_path, target_files=None):
    """读取测试文档
    
    Args:
        docs_path: 文档所在目录
        target_files: 指定要读取的文件名列表，为None或空列表则读取所有文件
    
    Returns:
        文档列表
    """
    documents = []
    
    # 检查目标文件列表
    has_target_files = target_files and len(target_files) > 0
    
    # 读取文件
    for file in os.listdir(docs_path):
        # 如果指定了目标文件，且当前文件不在目标列表中，则跳过
        if has_target_files and file not in target_files:
            continue
            
        file_path = os.path.join(docs_path, file)
        if os.path.isfile(file_path):
            try:
                content = read_document(file_path)
                if content:
                    documents.append({
                        "filename": file,
                        "path": file_path,
                        "content": content
                    })
                    logger.info(f"成功读取文档: {file}")
            except Exception as e:
                logger.error(f"读取文档 {file} 失败: {str(e)}")
                traceback.print_exc()
    return documents

def generate_document_summary(document, params):
    """生成文档摘要，不依赖Flask上下文"""
    try:
        # 构建参数
        temperature = params["temperature"]
        top_p = params["top_p"]
        top_k = params["top_k"]
        target_language = "zh"  # 固定使用中文
        summary_length = "medium"  # 固定使用中等长度
        
        # 获取输入文本
        input_text = document["content"]
        
        # 生成关键词
        try:
            keyword_list = generate_keywords_with_model(input_text[:3000], target_language)
            keywords = '|'.join(keyword_list)
            logger.info(f"生成的关键词: {keywords}")
        except Exception as e:
            logger.error(f"生成关键词失败: {str(e)}")
            keyword_list = get_default_keywords(target_language)
            keywords = '|'.join(keyword_list)
        
        # 摘要长度映射
        summary_length_map = {
            'very_short': 200,
            'medium': 500,
            'long': 2000,
            'very_long': 5000
        }
        target_word_count = summary_length_map.get(summary_length, 500)
        
        # 风格描述
        style_descriptions = {
            'basic': '以客观清晰的方式呈现信息',
            'academic': '使用学术风格，包含专业术语和严谨结构',
            'business': '聚焦商业价值和应用，使用商务语言',
            'technical': '详细描述技术细节和实现方法',
            'creative': '使用生动形象的语言，有故事性和感染力',
            'journalistic': '采用新闻报道风格，突出5W1H要素'
        }
        
        # 输出格式描述
        format_descriptions = {
            'paragraph': '连续段落式',
            'bullet': '要点列表式',
            'section': '带小标题的分节式',
            'narrative': '叙述式',
            'comparative': '对比式',
            'analytical': '分析式'
        }
        
        # 专业程度描述
        expertise_descriptions = {
            'introductory': '入门级，适合初学者',
            'intermediate': '中级，适合有基础的读者',
            'advanced': '高级，适合专业人士',
            'expert': '专家级，使用领域专业术语',
            'deductive': '演绎推理式，从原理到应用',
            'inductive': '归纳推理式，从案例到原理'
        }
        
        # 构建提示词
        summary_prompt = f"""你是一个专业的文档摘要专家。请严格按照思维链方法分析文档并生成高质量{target_language}摘要。

[思维链步骤]
1. 分析：仔细阅读文档，确定主题、目的和主要论点
2. 提取：识别核心概念、关键信息点和重要论述
3. 判断类型：确定文档是学术论文、研究报告、技术文档还是一般文章
4. 分类整理：按照合适的结构组织内容
5. 提炼：从每个部分提取最具代表性的内容
6. 关键词确认：验证已提取的关键词是否准确反映文档核心内容
7. 语言转换：将内容完全转换为{target_language}，保持专业术语的准确性
8. 整合：按照用户指定的参数要求生成最终摘要

[语言控制要求]
1. 摘要必须完全使用{target_language}，不得混合其他语言
2. 遵循{target_language}的语法规则和表达习惯
3. 专业术语需要使用{target_language}的对应表达方式
4. 确保摘要流畅自然，符合{target_language}的阅读习惯
5. 多次检查确保没有混入其他语言的词汇或表达

[输出格式要求]
1. 首先输出 [KEYWORDS] 标记
2. 在其下方输出4个关键词，用竖线(|)分隔
3. 然后输出 [SUMMARY] 标记
4. 最后输出摘要正文（全{target_language}）

[关键词]
{keywords}

[用户指定参数]
摘要长度：{target_word_count}字（±5%）
目标语言：{target_language}
temperature: {temperature}
top_p: {top_p}
top_k: {top_k}

[摘要结构要求]
1. 引言部分（约15%）：
   - 概述文档主题和背景
   - 点明核心问题或目的
   
2. 主体部分（约60%）：
   - 按照重要性组织关键信息点
   - 每个关键点需包含至少一个具体案例或数据支持
   
3. 结论部分（约25%）：
   - 总结文档的主要发现或结论
   - 如果适用，提供实践建议或未来展望

[内容质量要求]
1. 确保摘要总长度为{target_word_count}字（±5%）
2. 保持客观性和准确性，不添加原文中不存在的内容
3. 适当引用原文中的关键数据和证据支持观点
4. 确保整个摘要100%使用{target_language}，不混入其他语言

[原文内容]
{input_text}

请按照以上步骤和要求生成摘要，确保摘要的长度、风格、格式和内容符合用户指定的所有参数，并且严格使用{target_language}。
"""
        
        # 创建Ollama客户端
        client = Client(host='http://localhost:11434')
        
        # 使用Ollama生成摘要
        logger.info("正在生成摘要...")
        response = client.generate(
            model="huihui_ai/qwen2.5-1m-abliterated",
            prompt=summary_prompt,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        
        # 解析响应
        summary_text = response['response']
        logger.info(f"生成摘要完成，长度: {len(summary_text)}字")
        
        return summary_text
    except Exception as e:
        logger.error(f"生成摘要失败: {str(e)}")
        traceback.print_exc()
        return None

def evaluate_summary(original_text, summary_text):
    """使用大模型评估摘要质量"""
    try:
        # 构造评估提示词
        prompt = EVALUATION_PROMPT.format(
            original_text=original_text,
            summary_text=summary_text
        )
        
        # 调用API进行评估（使用流式模式）
        full_response = ""
        try:
            response_stream = openai_client.chat.completions.create(
                model="qwen3-235b-a22b",  # 使用阿里云千问模型
                messages=[
                    {"role": "system", "content": "你是一个专业的文档摘要评估专家，擅长使用思维链方法进行客观评估。你的评估必须保持高度一致性，对相同内容的评分应当基于相同标准。请遵循提供的评分方法，逐步分析，确保每次评分都是可重现的。在评估过程中，请首先分析原文和摘要的关系，然后明确列出评分理由，最后计算分数。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 使用更低温度确保评估的一致性
                stream=True,  # 启用流式响应
            )
            
            # 收集流式响应
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    
            logger.info("成功使用流式模式获取评估结果")
            
        except Exception as e:
            logger.error(f"流式评估失败，尝试使用本地模型: {str(e)}")
            # 如果阿里云API失败，使用本地Ollama模型评估
            try:
                client = Client(host='http://localhost:11434')
                ollama_prompt = f"""你是一个专业文档评估专家，擅长使用思维链方法进行客观评估。你需要基于以下原文和摘要进行评估：
                
                {prompt}
                
                请严格按照思维链步骤进行分析，确保评估过程客观、可重现且一致。完成分析后，提供评分并以JSON格式返回结果。"""
                
                ollama_resp = client.generate(
                    model="huihui_ai/qwen2.5-1m-abliterated",
                    prompt=ollama_prompt,
                    options={
                        "temperature": 0.1
                    }
                )
                full_response = ollama_resp['response']
                logger.info("成功使用本地模型获取评估结果")
            except Exception as local_err:
                logger.error(f"本地模型评估也失败: {str(local_err)}")
                # 使用默认评分
                return {
                    "信息准确性": 0.5,
                    "信息召回": 0.5,
                    "信息冗余": 0.5,
                    "信息质量": 0.5,
                    "总分": 2.0,
                    "分析": "自动评估失败，使用默认评分。"
                }
        
        # 提取JSON评分结果
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, full_response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            evaluation = json.loads(json_str)
        else:
            # 尝试直接解析整个文本
            try:
                # 查找可能的JSON部分
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = full_response[json_start:json_end]
                    evaluation = json.loads(json_str)
                else:
                    raise ValueError("无法找到JSON部分")
            except:
                logger.error(f"无法解析评估结果JSON: {full_response[:200]}...")
                # 使用默认评分
                evaluation = {
                    "信息准确性": 0.5,
                    "信息召回": 0.5,
                    "信息冗余": 0.5,
                    "信息质量": 0.5,
                    "总分": 2.0,
                    "分析": "无法解析评估结果，使用默认评分。"
                }
        
        return evaluation
    except Exception as e:
        logger.error(f"评估摘要失败: {str(e)}")
        traceback.print_exc()
        return {
            "信息准确性": 0.5,
            "信息召回": 0.5,
            "信息冗余": 0.5, 
            "信息质量": 0.5,
            "总分": 2.0,
            "分析": f"评估过程出错: {str(e)}"
        }

def process_document(document, config):
    """处理单个文档的评估"""
    try:
        logger.info(f"正在使用参数配置 '{config['name']}' 处理文档 {document['filename']}")
        
        # 生成摘要
        try:
            summary = generate_document_summary(document, config)
            if not summary:
                logger.error(f"文档 {document['filename']} 使用参数 '{config['name']}' 生成摘要失败")
                return None
                
            # 规范化摘要格式，提取正文部分
            # 寻找[SUMMARY]标记后的内容作为实际摘要
            if "[SUMMARY]" in summary:
                summary = summary.split("[SUMMARY]")[1].strip()
            # 移除可能的[KEYWORDS]部分
            if "[KEYWORDS]" in summary:
                summary = summary.split("[KEYWORDS]")[0].strip()
                
            logger.info(f"规范化后的摘要长度: {len(summary)}字")
        except Exception as e:
            logger.error(f"文档 {document['filename']} 生成摘要时出错: {str(e)}")
            return None
            
        # 评估摘要
        try:
            evaluation = evaluate_summary(document["content"], summary)
            if not evaluation:
                logger.error(f"文档 {document['filename']} 使用参数 '{config['name']}' 评估摘要失败")
                return None
        except Exception as e:
            logger.error(f"文档 {document['filename']} 评估摘要时出错: {str(e)}")
            return None
        
        # 返回结果
        result = {
            "filename": document["filename"],
            "config_name": config["name"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "信息准确性": evaluation["信息准确性"],
            "信息召回": evaluation["信息召回"],
            "信息冗余": evaluation["信息冗余"],
            "信息质量": evaluation["信息质量"],
            "总分": evaluation["总分"],
            "分析": evaluation["分析"]
        }
        
        logger.info(f"文档 {document['filename']} 使用参数 '{config['name']}' 的评估完成，总分: {evaluation['总分']}")
        return result
    except Exception as e:
        logger.error(f"处理文档 {document['filename']} 失败: {str(e)}")
        traceback.print_exc()
        return None

def generate_report(results):
    """生成评估报告"""
    try:
        # 确保有结果可以处理
        if not results or len(results) == 0:
            logger.error("没有评估结果可供处理，无法生成报告")
            return
            
        # 确保结果文件夹存在
        if not os.path.exists(RESULTS_FOLDER):
            os.makedirs(RESULTS_FOLDER)
            logger.info(f"创建结果文件夹: {RESULTS_FOLDER}")
            
        # 转换为DataFrame便于分析
        df = pd.DataFrame(results)
        
        # 检查有多少个不同的参数配置
        unique_configs = df['config_name'].unique()
        logger.info(f"结果中包含 {len(unique_configs)} 个不同的参数配置")
        
        # 计算每个参数配置的平均分
        avg_scores = df.groupby('config_name').agg({
            '信息准确性': 'mean',
            '信息召回': 'mean',
            '信息冗余': 'mean',
            '信息质量': 'mean',
            '总分': 'mean'
        }).reset_index()
        
        # 文件名生成（带时间戳或固定名称）
        if OVERWRITE_RESULTS:
            # 固定文件名，总是覆盖
            csv_path = os.path.join(RESULTS_FOLDER, "summary_evaluation.csv")
            md_path = os.path.join(RESULTS_FOLDER, "summary_evaluation_report.md")
            total_chart_path = os.path.join(RESULTS_FOLDER, "summary_evaluation_total.png")
            metrics_chart_path = os.path.join(RESULTS_FOLDER, "summary_evaluation_metrics.png")
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 仅用于报告内容
        else:
            # 使用时间戳创建新文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(RESULTS_FOLDER, f"summary_evaluation_{timestamp}.csv")
            md_path = os.path.join(RESULTS_FOLDER, f"summary_evaluation_report_{timestamp}.md")
            total_chart_path = os.path.join(RESULTS_FOLDER, f'summary_evaluation_total_{timestamp}.png')
            metrics_chart_path = os.path.join(RESULTS_FOLDER, f'summary_evaluation_metrics_{timestamp}.png')
        
        # 保存为CSV备份
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"详细评估结果已保存至: {csv_path}")
        
        # 生成Markdown报告
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 文档摘要参数评估报告\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"测试文档路径: {TEST_DOCS_PATH}\n\n")
            f.write(f"测试文档数量: {df['filename'].nunique()}\n\n")
            
            # 若有文档评估失败，在报告中注明
            total_possible_evaluations = len(PARAMETER_CONFIGS) * df['filename'].nunique()
            if len(results) < total_possible_evaluations:
                f.write(f"**注意**: 部分文档评估失败，共完成 {len(results)}/{total_possible_evaluations} 个评估组合。\n\n")
            
            # 参数配置概览
            f.write("## 参数配置\n\n")
            f.write("| 配置名称 | Temperature | Top P | Top K |\n")
            f.write("|---------|-------------|-------|-------|\n")
            for config in PARAMETER_CONFIGS:
                f.write(f"| {config['name']} | {config['temperature']} | {config['top_p']} | {config['top_k']} |\n")
            
            f.write("\n## 评估指标说明\n\n")
            f.write("- **信息准确性**: 检查是否有数字错误、主体错误、事实错误等\n")
            f.write("- **信息召回**: 摘要包含原文内容的比例\n")
            f.write("- **信息冗余**: 是否包含不属于原文的内容\n")
            f.write("- **信息质量**: 可读性问题与重复性问题\n\n")
            
            # 平均分表格
            f.write("## 参数配置评分汇总\n\n")
            f.write("| 配置名称 | 信息准确性 | 信息召回 | 信息冗余 | 信息质量 | 总分 | 成功评估数 |\n")
            f.write("|---------|------------|---------|----------|----------|------|------------|\n")
            
            # 添加每个配置的成功评估数
            config_counts = df['config_name'].value_counts().to_dict()
            
            for _, row in avg_scores.iterrows():
                config_name = row['config_name']
                success_count = config_counts.get(config_name, 0)
                f.write(f"| {config_name} | {row['信息准确性']:.2f} | {row['信息召回']:.2f} | ")
                f.write(f"{row['信息冗余']:.2f} | {row['信息质量']:.2f} | {row['总分']:.2f} | {success_count} |\n")
            
            # 每个文档的详细结果
            f.write("\n## 文档详细评分\n\n")
            
            # 按文档分组
            for filename in sorted(df['filename'].unique()):
                f.write(f"### 文档: {filename}\n\n")
                doc_df = df[df['filename'] == filename]
                
                f.write("| 配置名称 | 信息准确性 | 信息召回 | 信息冗余 | 信息质量 | 总分 |\n")
                f.write("|---------|------------|---------|----------|----------|------|\n")
                
                for _, row in doc_df.iterrows():
                    f.write(f"| {row['config_name']} | {row['信息准确性']:.2f} | {row['信息召回']:.2f} | ")
                    f.write(f"{row['信息冗余']:.2f} | {row['信息质量']:.2f} | {row['总分']:.2f} |\n")
                
                f.write("\n")
            
            # 找出最佳配置 (如果有足够数据)
            if not avg_scores.empty:
                best_config = avg_scores.loc[avg_scores['总分'].idxmax()]
                f.write("\n## 评估结论\n\n")
                f.write(f"综合评分最高的参数配置为: **{best_config['config_name']}**，平均得分: {best_config['总分']:.2f}\n\n")
                f.write("具体参数:\n\n")
                
                # 查找对应的配置参数
                for config in PARAMETER_CONFIGS:
                    if config['name'] == best_config['config_name']:
                        f.write(f"- Temperature: {config['temperature']}\n")
                        f.write(f"- Top P: {config['top_p']}\n")
                        f.write(f"- Top K: {config['top_k']}\n")
                        break
            else:
                f.write("\n## 评估结论\n\n")
                f.write("由于评估样本不足，无法得出最佳配置结论。\n")
        
        logger.info(f"评估报告已生成: {md_path}")
        print(f"评估报告已生成: {md_path}")
        
        # 只有在有足够数据时才绘制图表
        if len(unique_configs) > 1 and not avg_scores.empty:
            # 绘制评分图表
            plot_evaluation_results(avg_scores, total_chart_path, metrics_chart_path)
        else:
            logger.warning("配置数量不足，跳过图表生成")
        
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}")
        traceback.print_exc()

def plot_evaluation_results(avg_scores, total_chart_path, metrics_chart_path):
    """绘制评估结果图表"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制总分对比图
        plt.figure(figsize=(10, 6))
        plt.bar(avg_scores['config_name'], avg_scores['总分'], color='skyblue')
        plt.xlabel('参数配置')
        plt.ylabel('平均总分')
        plt.title('不同参数配置的摘要总分对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(total_chart_path)
        
        # 绘制各指标对比图
        metrics = ['信息准确性', '信息召回', '信息冗余', '信息质量']
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(avg_scores['config_name']))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width - 0.3, avg_scores[metric], width, label=metric)
        
        plt.xlabel('参数配置')
        plt.ylabel('平均分')
        plt.title('不同参数配置的各指标对比')
        plt.xticks(x, avg_scores['config_name'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(metrics_chart_path)
        
        logger.info("评估结果图表已生成")
    except Exception as e:
        logger.error(f"生成图表失败: {str(e)}")
        traceback.print_exc()

def run_evaluation(target_files=None):
    """运行完整评估流程
    
    Args:
        target_files: 要评估的文件名列表，为None则使用全局TARGET_FILES
    """
    try:
        # 使用参数指定的文件列表，或全局设置
        files_to_evaluate = target_files if target_files is not None else TARGET_FILES
        
        # 读取测试文档
        logger.info(f"开始读取测试文档，路径: {TEST_DOCS_PATH}")
        if files_to_evaluate:
            logger.info(f"指定要评估的文件: {files_to_evaluate}")
            
        documents = read_test_documents(TEST_DOCS_PATH, files_to_evaluate)
        logger.info(f"共读取 {len(documents)} 个测试文档")
        
        if not documents:
            logger.error("未找到可用的测试文档，评估终止")
            return
            
        # 存储所有评估结果
        all_results = []
        # 记录失败的文档
        failed_documents = []
        
        # 使用线程池加速处理
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            futures_map = {}  # 用于追踪每个future对应的文档和配置
            
            # 提交所有任务
            for doc in documents:
                for config in PARAMETER_CONFIGS:
                    future = executor.submit(process_document, doc, config)
                    futures.append(future)
                    futures_map[future] = (doc['filename'], config['name'])
            
            # 收集结果
            for future in tqdm(futures, desc="评估进度"):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                    else:
                        doc_name, config_name = futures_map[future]
                        failed_documents.append(f"{doc_name} (配置: {config_name})")
                        logger.warning(f"文档 {doc_name} 使用配置 {config_name} 的评估失败，已跳过")
                except Exception as e:
                    doc_name, config_name = futures_map[future]
                    failed_documents.append(f"{doc_name} (配置: {config_name})")
                    logger.error(f"处理文档 {doc_name} 使用配置 {config_name} 时发生错误: {str(e)}")
        
        # 打印失败的文档信息
        if failed_documents:
            logger.warning(f"评估过程中有 {len(failed_documents)} 个文档-配置组合失败:")
            for failed_doc in failed_documents:
                logger.warning(f"  - {failed_doc}")
            print(f"注意: {len(failed_documents)} 个文档-配置组合在评估过程中被跳过")
        
        # 生成结果报告（即使有失败的文档也继续生成报告）
        if all_results:
            generate_report(all_results)
        else:
            logger.error("所有评估都失败，无法生成报告")
            print("所有评估都失败，无法生成报告")
    except Exception as e:
        logger.error(f"评估过程失败: {str(e)}")
        traceback.print_exc()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='文档摘要参数评估工具')
    parser.add_argument('--files', nargs='+', help='要评估的文件名列表，未指定则评估所有文件')
    parser.add_argument('--overwrite', action='store_true', help='覆盖旧的评估结果，而不是生成新文件')
    parser.add_argument('--keep-all', action='store_true', help='保留所有评估结果，生成带时间戳的新文件')
    return parser.parse_args()

#--------- 主程序 ---------#

if __name__ == "__main__":
    try:
        # 导入tqdm模块 (仅在需要时导入)
        from tqdm import tqdm
        
        args = parse_arguments()
        
        # 设置是否覆盖旧结果
        if args.overwrite:
            OVERWRITE_RESULTS = True
        elif args.keep_all:
            OVERWRITE_RESULTS = False
        # 否则使用默认设置
        
        print("开始摘要参数评估...")
        print(f"结果模式: {'覆盖模式' if OVERWRITE_RESULTS else '保留所有结果'}")
        
        # 检查是否通过命令行指定了文件
        if args.files:
            print(f"将仅评估以下文件: {args.files}")
            run_evaluation(args.files)
        else:
            # 如果未指定文件，且存在全局TARGET_FILES，则使用它
            if TARGET_FILES:
                print(f"将仅评估以下文件: {TARGET_FILES}")
            else:
                print(f"将评估 {TEST_DOCS_PATH} 下的所有文件")
            run_evaluation()
            
        print("评估完成！")
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 