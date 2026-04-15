# 智能问数系统 (Smart Analytics)

基于 **LangGraph + MCP + vLLM + FastAPI** 的现代化智能数据分析与可视化平台。

## 🌟 核心特性

- **自然语言查询**: 使用中文提问，AI 自动生成 SQL 并查询数据库
- **智能可视化**: 自动推荐并生成交互式图表 (Plotly)
- **高性能架构**: FastAPI 异步后端 + vLLM 高速推理
- **MCP 协议**: 标准化数据库和可视化工具调用
- **现代化前端**: 响应式设计，美观易用

## 🏗️ 技术架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   用户界面   │────▶│  FastAPI     │────▶│  LangGraph  │
│  (Vue3/HTML)│     │   后端服务    │     │   Agent     │
└─────────────┘     └──────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                   ┌──────────────┐     ┌─────────────┐
                   │   MCP Server │     │    vLLM     │
                   │  (数据库)     │     │  (LLM 服务)  │
                   └──────────────┘     └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   MySQL DB   │
                   └──────────────┘
```

## 📁 项目结构

```
smart_analytics/
├── backend/           # FastAPI 后端
│   ├── app.py        # 主应用入口
│   └── mcp_client.py # MCP 客户端
├── frontend/         # 现代化前端
│   ├── index.html    # 主页面
│   ├── styles.css    # 样式文件
│   └── app.js        # 交互逻辑
├── langgraph_agent/  # LangGraph Agent
│   ├── graph.py      # 状态机定义
│   └── nodes.py      # 节点处理逻辑
├── mcp_servers/      # MCP 服务器
│   ├── db_server.py  # 数据库服务
│   └── viz_server.py # 可视化服务
├── llm_service/      # vLLM 部署配置
│   └── docker-compose.yml
├── config/           # 配置文件
│   └── settings.py
├── requirements.txt  # Python 依赖
└── README.md         # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd smart_analytics

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库和 LLM 信息
```

### 3. 启动 vLLM 服务

```bash
cd llm_service
docker-compose up -d
```

### 4. 启动 MCP 服务器

```bash
# 终端 1: 数据库 MCP 服务器
python mcp_servers/db_server.py

# 终端 2: 可视化 MCP 服务器
python mcp_servers/viz_server.py
```

### 5. 启动 FastAPI 后端

```bash
cd backend
python app.py
# 或 uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. 访问系统

打开浏览器访问：`http://localhost:8000`

## 📊 功能演示

### 示例查询

1. **基础统计**
   ```
   统计每个用户的文档数量，并按降序排列
   ```

2. **时间范围查询**
   ```
   显示最近创建的 10 个文档及其摘要
   ```

3. **分布分析**
   ```
   分析文档类型的分布情况，用饼图展示
   ```

4. **趋势分析**
   ```
   显示近 30 天文档创建数量的趋势
   ```

## ⚙️ 配置说明

### 环境变量 (.env)

```ini
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password

# vLLM 配置
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=qwen2.5-coder:7b

# FastAPI 配置
FLASK_HOST=0.0.0.0
FLASK_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# 日志级别
LOG_LEVEL=INFO
```

## 🔧 开发指南

### 添加新的 MCP 工具

1. 在 `mcp_servers/` 创建新的服务器
2. 实现标准的 MCP 协议接口
3. 在 LangGraph Agent 中注册工具

### 自定义图表类型

修改 `mcp_servers/viz_server.py` 中的图表推荐逻辑

### 调整 Agent 流程

编辑 `langgraph_agent/graph.py` 中的状态机定义

## 📈 性能优化建议

1. **vLLM 配置**: 根据 GPU 显存调整 batch size
2. **数据库索引**: 为常用查询字段添加索引
3. **缓存层**: 添加 Redis 缓存频繁查询结果
4. **连接池**: 调整数据库连接池大小

## 🛡️ 安全注意事项

- MCP 数据库服务器使用只读账号
- 启用 CORS 白名单限制
- 生产环境使用 HTTPS
- 定期更新依赖包

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!
