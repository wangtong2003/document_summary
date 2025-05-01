import os
from openai import OpenAI


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-c278401b29864808b81c3d7c841300e1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 使用流式输出模式
completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen3-235b-a22b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    # 开启流式输出
    stream=True,
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 取消注释以下行可禁用思考过程
    # extra_body={"enable_thinking": False},
)

# 处理流式响应
full_response = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        full_response += content
        print(content, end="", flush=True)

print("\n\n完整响应:")
print(full_response)