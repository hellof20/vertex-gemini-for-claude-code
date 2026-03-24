# Anthropic to Vertex AI Gemini Proxy

这是一个基于 FastAPI 的代理服务器，旨在将 Anthropic 格式的 API 请求转换为 Google Vertex AI Gemini API 格式。它支持普通对话和流式输出，并处理了工具调用（Tool Use）的转换。

## 功能特性

- **协议转换**：完美支持 Anthropic 消息格式到 Gemini 内容格式的转换。
- **流式输出**：支持 Server-Sent Events (SSE) 流式响应。
- **工具调用**：支持 Anthropic 格式的工具定义和调用转换。
- **模型映射**：可以通过 `config.yaml` 自定义 Anthropic 模型到 Gemini 模型的映射。
- **完善的日志统计**：自动记录 Token 消耗（支持 Cache 和 Thinking Tokens），并对 API 错误信息进行了易读性优化。

## 快速开始

### 1. 安装依赖

确保您已安装 Python 3.9+，然后安装所需依赖：

```bash
pip install fastapi uvicorn google-genai pyyaml
```

### 2. 配置说明

项目支持通过 `config.yaml` 或环境变量进行配置：

- `VERTEX_PROJECT_ID`: 您的 Google Cloud 项目 ID。
- `VERTEX_REGION`: Vertex AI 区域（默认为 `global`）。
- `SERVER_PORT`: 代理服务器运行端口（默认为 `8765`）。

### 3. 认证配置 (Authentication)

本项目支持多种 Google Cloud 认证方式：

#### 方式一：API Key (推荐，最简便)

1. [在 Google Cloud 控制台创建 API Key](https://console.cloud.google.com/apis/credentials)。
2. 确保该 Key 拥有访问 Vertex AI 的权限。
3. 设置环境变量（支持以下任一变量名）：
   ```bash
   export VERTEX_API_KEY="您的_API_KEY"
   # 或者
   export GOOGLE_CLOUD_API_KEY="您的_API_KEY"
   ```

#### 方式二：本地开发环境 (ADC)

推荐使用 Google Cloud CLI 进行交互式登录：

```bash
gcloud auth application-default login
```

#### 方式三：服务器或 Docker 环境 (Service Account)

建议使用服务账号 (Service Account) 并通过环境变量指定凭据路径：

1. [创建服务账号](https://console.cloud.google.com/iam-admin/serviceaccounts) 并下载 JSON 密钥。
2. 为该账号分配 **Vertex AI User** 角色。
3. 设置环境变量：
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
   ```

### 4. 运行服务器

```bash
python server.py
```

### 5. 使用代理

启动后，您可以将 Anthropic 客户端的 Base URL 指向本代理：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8765
```

## 项目结构

- `server.py`: FastAPI 应用入口，处理路由和请求转发。
- `converter.py`: 核心转换逻辑，负责 Anthropic 和 Gemini 之间的消息格式互转。
- `config.py`: 配置加载逻辑。
- `config.yaml`: 默认配置文件。

## 注意事项

- 运行前请确保已配置好 Google Cloud 认证环境（如 `gcloud auth application-default login`）。
- 本项目暂不保证所有 Anthropic 特性的 100% 兼容，仅涵盖核心对话功能。
