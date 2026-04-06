# 🧠 Thinker – 批判性思维分析器

基于 Vincent Ruggiero《Beyond Feelings: A Guide to Critical Thinking》的核心框架，帮助你对任何一段文本进行四步批判性分析。

## ✨ 功能

| 步骤 | 说明 |
|------|------|
| **1. 识别事实、情绪与假设** | 将文本拆解为可验证的事实、情绪化表达和隐含假设 |
| **2. 检测思维误区** | 对照 18 种常见思维谬误逐一排查 |
| **3. 提出三种可能的解释** | 给出至少三种不同的、合理的解读 |
| **4. 给出更理性的结论** | 综合分析，输出基于证据的平衡结论 |

- 🌐 支持**中文**和**英文**界面及分析输出
- 📖 **原文阅读** — 内置 PDF 章节目录，点击即可查看对应章节内容（支持文字 PDF 和扫描版图片渲染）
- 📝 **书本问答** — 基于书籍内容提问，AI 根据原文总结回答，并标注相关章节
- 💾 所有分析历史自动保存到 **SQLite**，可随时回看和删除
- 🔌 LLM Provider **抽象接口**，支持 OpenAI / DeepSeek / 智谱 GLM / Ollama 等

## 📦 安装

```bash
# 1. 克隆项目
cd thinker

# 2. 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置

设置 OpenAI API Key（二选一）：

```bash
# 方式 A：环境变量
export OPENAI_API_KEY="sk-..."

# 方式 B：在 Streamlit 侧边栏中直接输入
```

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI API 密钥 |
| `OPENAI_MODEL` | `gpt-4o` | 模型名称 |
| `OPENAI_BASE_URL` | `None` | 兼容 API 端点（Azure、vLLM 等） |

## 🚀 运行

```bash
streamlit run main.py
```

浏览器会自动打开 `http://localhost:8501`。

## 📁 项目结构

```
thinker/
├── main.py                     # Streamlit 入口（3个Tab：分析/阅读/问答）
├── config.py                   # 配置项
├── requirements.txt
├── README.md
├── analyzer/
│   ├── __init__.py
│   ├── models.py               # Pydantic 数据模型
│   ├── prompts.py              # 中英文 Prompt 模板
│   ├── engine.py               # ThinkerEngine 分析引擎
│   └── providers/
│       ├── __init__.py         # BaseProvider 抽象基类
│       ├── openai_provider.py  # OpenAI 实现（含 complete_text）
│       └── ollama_provider.py  # Ollama 占位
├── book/
│   ├── __init__.py
│   ├── reader.py               # PDF 章节提取 + 全文提取
│   └── *.pdf                   # Beyond Feelings PDF 文件
├── db/
│   ├── __init__.py
│   ├── database.py             # SQLite CRUD
│   └── thinker.db              # 运行时自动创建
└── i18n/
    ├── zh.json                 # 中文界面文案
    └── en.json                 # 英文界面文案
```

## 📖 理论背景

《Beyond Feelings》将思维过程分为三个层次：
1. **感觉（Feelings）** — 未经审视的直觉反应
2. **观点（Opinions）** — 基于感觉形成的主观判断
3. **理性思考（Critical Thinking）** — 超越感觉，用系统方法评估证据、识别谬误、得出合理结论

本工具将这一框架自动化，辅助用户从"感觉驱动"转向"证据驱动"的思考方式。

## 📄 License

MIT
