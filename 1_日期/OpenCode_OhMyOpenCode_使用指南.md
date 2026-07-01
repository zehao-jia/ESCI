# OpenCode + OhMyOpenCode 使用指南

> 基于本机（Windows）实际安装环境整理 · v1.14.48

---

## 目录

1. [概述](#1-概述)
2. [安装与升级](#2-安装与升级)
3. [核心命令](#3-核心命令)
4. [模型提供商与配置](#4-模型提供商与配置)
5. [OhMyOpenCode 多智能体系统](#5-ohmyopencode-多智能体系统)
6. [Skills（技能系统）](#6-skills技能系统)
7. [会话管理](#7-会话管理)
8. [MCP 与插件](#8-mcp-与插件)
9. [Profile 切换](#9-profile-切换)
10. [最佳实践与工作流](#10-最佳实践与工作流)
11. [常见问题排查](#11-常见问题排查)

---

## 1. 概述

### 1.1 什么是 OpenCode？

**OpenCode** 是一个终端原生的 AI 编码助手（TUI - Terminal UI），类似 Claude Code 的开源替代。它运行在终端中，直接与你的代码仓库交互，通过 AI 模型驱动完成代码编写、调试、重构、搜索等开发任务。

- **官网**: [opencode.ai](https://opencode.ai)
- **当前版本**: 1.14.48
- **运行方式**: Node.js 应用，通过 npm 全局安装
- **平台**: Windows / macOS / Linux

### 1.2 什么是 OhMyOpenCode？

**OhMyOpenCode**（原 OhMyOpenAgent）是一个 OpenCode 的**增强配置框架**，由 `code-yeongyu` 开发。它在 OpenCode 的基础上增加了：

- **多智能体系统** — Sisyphus、Oracle、Librarian 等专业分工的 AI Agent
- **分类任务系统** — visual-engineering、ultrabrain、deep 等任务类别，每类使用最适合的模型
- **Skills 技能系统** — 可插拔的专业技能模块
- **Fallback 机制** — 模型故障时自动降级到备选模型
- **Profile 切换** — 在不同模型提供商之间一键切换

> GitHub: [github.com/code-yeongyu/oh-my-openagent](https://github.com/code-yeongyu/oh-my-openagent)

### 1.3 架构关系

```
┌─────────────────────────────────────────────────┐
│                  OpenCode                         │
│  (TUI 界面 / 命令调度 / 会话管理 / 文件交互)       │
├─────────────────────────────────────────────────┤
│              OhMyOpenCode (oh-my-openagent.json)  │
│  (Agent 定义 / 分类映射 / 模型选择 / Fallback)    │
├─────────────────────────────────────────────────┤
│          Skills 技能系统 (skills/ 目录)            │
│  (专业指令 / 工作流模板 / 工具映射)               │
├─────────────────────────────────────────────────┤
│          模型提供商 (API 代理层)                   │
│  jiekou.ai / volcengine / anthropic / openai …    │
└─────────────────────────────────────────────────┘
```

---

## 2. 安装与升级

### 2.1 安装 OpenCode

```powershell
npm install -g opencode-ai
```

安装后，启动命令：
```powershell
opencode
```

### 2.2 升级

```powershell
# 升级到最新版
opencode upgrade

# 升级到指定版本
opencode upgrade 1.14.48
```

### 2.3 卸载

```powershell
opencode uninstall
```

### 2.4 验证安装

```powershell
opencode --version   # → 1.14.48
opencode --help      # → 显示所有命令
```

---

## 3. 核心命令

### 3.1 启动与运行

```powershell
# 在当前目录启动 TUI
opencode

# 在指定项目目录启动
opencode /path/to/project

# 直接运行一条消息（非交互模式）
opencode run "解释一下这个项目的架构"

# 启动 Web UI
opencode web

# 启动无头服务器
opencode serve
```

### 3.2 会话管理

```powershell
# 列出所有会话
opencode session list

# 查看特定会话
opencode session show <session-id>

# 导出会话为 JSON
opencode export <session-id>

# 导入会话
opencode import <file.json>
```

### 3.3 提供者与模型

```powershell
# 管理 API 提供者（交互式）
opencode providers
# 或
opencode auth

# 列出所有可用模型
opencode models

# 查看某提供者的模型
opencode models anthropic
```

### 3.4 插件与 MCP

```powershell
# 安装插件
opencode plugin <npm-package-name>

# 管理 MCP 服务器
opencode mcp
```

### 3.5 调试与统计

```powershell
# 调试工具
opencode debug

# Token 使用统计
opencode stats

# 数据库工具
opencode db

# 打印日志
opencode --print-logs --log-level DEBUG
```

### 3.6 GitHub 集成

```powershell
# 管理 GitHub Agent
opencode github

# 基于 PR 启动工作会话
opencode pr <pr-number>
```

### 3.7 其他

```powershell
# ACP (Agent Client Protocol) 服务器
opencode acp

# Shell 补全
opencode completion

# 附加到运行中的服务器
opencode attach <url>
```

---

## 4. 模型提供商与配置

### 4.1 主配置文件

**路径**: `C:\Users\<user>\.config\opencode\opencode.json`

本机配置了三个提供商，全部通过统一的 API 代理 `jiekou.ai` 访问：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "claude-sonnet-4-20250514",      // 默认模型
  "provider": {
    "anthropic": {
      "options": {
        "baseURL": "https://api.jiekou.ai/anthropic",
        "apiKey": "sk_G7..."
      },
      "models": {
        "claude-sonnet-4-20250514": {},
        "claude-opus-4-7": {},
        "claude-haiku-4-5-20251001": {}
      }
    },
    "openai": {
      "options": {
        "baseURL": "https://api.jiekou.ai/openai/v1",
        "apiKey": "sk_G7..."
      },
      "models": {
        "gpt-4o": {},
        "gpt-5.5": {},
        "o3": {},
        "deepseek/deepseek-v3-0324": {},
        "grok-3": {},
        "qwen/qwen3-235b-a22b-instruct-2507": {},
        "moonshotai/kimi-k2.5": {}
      }
    },
    "gemini": { ... },
    "deepseek": { ... }
  }
}
```

> **注意**: `opencode.json` 中的 `"model"` 字段仅决定**主会话**使用的模型。子 Agent 的模型由 `oh-my-openagent.json` 决定。

### 4.2 模型命名格式

OhMyOpenCode 中，模型名称的格式为 `provider/model-name`：

| 格式 | 示例 | 说明 |
|------|------|------|
| `anthropic/claude-opus-4-7` | Anthropic Claude Opus 4.7 | 通过 jiekou.ai 代理 |
| `openai/gpt-5.5` | OpenAI GPT-5.5 | 通过 jiekou.ai 代理 |
| `google/gemini-3.1-pro-preview` | Google Gemini 3.1 Pro | 通过 jiekou.ai 代理 |
| `volcengine-coding/deepseek-v3.2` | 火山引擎 DeepSeek V3.2 | 直连火山引擎 |

---

## 5. OhMyOpenCode 多智能体系统

### 5.1 智能体总览

核心配置文件：`C:\Users\<user>\.config\opencode\oh-my-openagent.json`

本系统定义了 **11 个专用智能体** 和 **8 个任务类别**：

#### 🧠 核心智能体（Agent）

| 智能体 | 模型 | 用途 | 说人话版 |
|--------|------|------|----------|
| **Sisyphus** | `claude-opus-4-7` | 主控 Agent，负责规划、决策、委派 | 你现在的角色，总管 |
| **Sisyphus-Junior** | `claude-sonnet-4-6` | 轻量级执行 Agent，按指令完成具体任务 | 小弟，干活的 |
| **Oracle** | `gpt-5.5` | 只读顾问，架构评审、复杂调试 | 老专家，只看不写 |
| **Librarian** | `gpt-5.4-mini-fast` | 外部文档检索、GitHub 搜索、库用法查询 | 图书管理员 |
| **Explore** | `gpt-5.4-mini-fast` | 代码库搜索、模式发现、文件定位 | 侦探，翻代码的 |
| **Metis** | `claude-opus-4-7` | 任务规划顾问，识别歧义和盲点 | 军师，事前分析 |
| **Momus** | `gpt-5.5` | 计划评审，检查完整性/可验证性 | 批评家，事后挑刺 |
| **Prometheus** | `claude-opus-4-7` | 高级 Planner，制定执行方案 | 战略规划师 |
| **Hephaestus** | `gpt-5.5` | 通用构建/实现 Agent | 工匠 |
| **Multimodal-Looker** | `gpt-5.5` | 图像/PDF 等多媒体文件分析 | 看图说话的 |
| **Atlas** | `claude-sonnet-4-6` | 代码库知识图谱维护 | 地图绘制员 |

#### 📂 任务类别（Category）

| 类别 | 模型 | 适用场景 |
|------|------|----------|
| **visual-engineering** | `gemini-3.1-pro-preview` | 前端/UI/UX/样式/动画 |
| **ultrabrain** | `gpt-5.5` | 硬核逻辑、算法、复杂推理 |
| **deep** | `gpt-5.5` | 深度研究 + 端到端实现 |
| **artistry** | `gemini-3.1-pro-preview` | 非常规问题、创造性解法 |
| **quick** | `gpt-5.4-mini` | 单文件修改、简单改动 |
| **unspecified-low** | `claude-sonnet-4-6` | 其他低难度任务 |
| **unspecified-high** | `claude-sonnet-4-6` | 其他高难度任务 |
| **writing** | `gemini-3-flash-preview` | 文档/写作 |

### 5.2 Fallback 机制

每个智能体和类别都配置了 **fallback_models**。当主模型不可用时自动降级：

```
示例: Oracle 的 fallback 链
gpt-5.5 (high) → gemini-3.1-pro-preview (high) → claude-opus-4-7 (max)
```

### 5.3 Variant（变体）说明

| Variant | 含义 |
|---------|------|
| `max` | 最强输出质量，最贵/token 最多 |
| `high` | 高质量，适度权衡 |
| `medium` | 平衡模式 |
| `xhigh` | 极度高质量（比 high 更高一级） |
| `mini` / `nano` | 轻量快速，适合简单任务 |
| `fast` | 快速模式，更低延迟 |

### 5.4 在 Prompt 中调用子 Agent

```typescript
// 方式一：按类别委托（推荐）
task(
  category="quick",           // 匹配任务类型
  load_skills=["caveman"],    // 加载技能
  prompt="修复这个 typo...",
  run_in_background=false     // 同步等待
)

// 方式二：直接指定子 Agent
task(
  subagent_type="explore",    // 直接指定 Agent
  load_skills=[],
  prompt="在 src/ 下搜索所有 API route 定义...",
  run_in_background=true      // 后台运行
)

// 方式三：继续之前的任务
task(
  task_id="ses_xxx",          // 保留之前 Agent 的完整上下文
  load_skills=[],
  prompt="修复之前发现的类型错误"
)
```

### 5.5 各 Agent 职责详解

#### Sisyphus（你正在对话的角色）
- **核心定位**: 主控调度 Agent
- **行为准则**:
  - 不独自工作，有专业 Agent 时优先委派
  - 不自作主张实现功能，除非用户明确要求
  - 任何实现前先做代码库调研
  - 完成后必须实际验证（非"应该能跑"）

#### Oracle
- **核心定位**: 只读顾问，高 IQ 推理
- **什么时候用**:
  - 复杂架构设计前
  - 2 次以上修复尝试失败
  - 不熟悉代码模式
  - 安全/性能评审
- **约束**: 只读，不写代码

#### Librarian
- **核心定位**: 外部资料检索
- **能力**: GitHub 搜索、Context7 文档查询、Web 搜索
- **什么时候用**:
  - 遇到不熟悉的库
  - 需要查官方文档
  - 要在开源项目里找实现参考

#### Explore
- **核心定位**: 代码库内部搜索
- **什么时候用**:
  - "这个功能在哪实现的？"
  - "类似的模式在别处怎么用的？"
  - 不熟悉的模块结构分析

#### Metis
- **核心定位**: 事前规划分析
- **什么时候用**:
  - 复杂任务需要先拆解
  - 需求含糊需要澄清
  - 需要识别可能出问题的地方

#### Momus
- **核心定位**: 事后质量评审
- **什么时候用**:
  - 实现完成后做质量检查
  - 计划评审
  - 查漏补缺

---

## 6. Skills（技能系统）

### 6.1 什么是 Skills？

Skills 是 OpenCode/OhMyOpenCode 的**可插拔指令模块**。每个 Skill 包含一套专业工作流，当任务匹配时自动加载。

**本机已安装的技能**（位于 `%USERPROFILE%\.config\opencode\skills\`）：

| 技能 | 用途 |
|------|------|
| **caveman** | 极简模式，减少 Token 消耗 ~75% |
| **paper-quick-read** | 学术论文快速阅读摘要生成 |
| **pdf-reader** | PDF 文件解析（调用本地 Python 脚本） |
| **planning-with-files-zh** | 基于文件的规划系统（task_plan.md / progress.md） |
| **ppt-image-first** | PPT 演示文稿规划与设计 |
| **using-superpowers** | 启动引导，教 AI 如何使用 Skills |

### 6.2 加载 Skill

在 Prompt 中主动引用：
```
请用 /caveman 模式回答
```

Agent 侧自动加载：
```typescript
// 委托任务时加载技能
task(
  category="visual-engineering",
  load_skills=["frontend-ui-ux"],  // 技能名称
  prompt="...",
  run_in_background=false
)
```

### 6.3 skill 工具的使用

```typescript
// 手动加载 skill 查看其指令
skill(name="caveman")
```

### 6.4 Skill 优先级

```
用户指令 > Skills > 系统默认 Prompt
```

如果 `CLAUDE.md` 说"不要用 TDD"，而某个 Skill 说"必须用 TDD"——以用户指令为准。

### 6.5 安装自定义 Skill

将 Skill 文件夹放入 `%USERPROFILE%\.config\opencode\skills\` 目录即可。每个 Skill 文件夹需要包含 `SKILL.md`（以及可选的 `CLAUDE_GLOBAL_INSTRUCTIONS.md`）。

---

## 7. 会话管理

### 7.1 会话概念

每个 `opencode` 启动实例创建一个**会话**。会话记录了：
- 所有消息历史
- AI 的 Todo 列表
- 文件修改记录
- Token 使用统计

### 7.2 会话命令

```powershell
# 列出所有会话
opencode session list

# 查看会话详情
opencode session show <session-id>

# 查看会话中的消息
opencode session read <session-id>

# 搜索会话内容
opencode session search "关键词"

# 导出会话
opencode export <session-id>

# 导入会话
opencode import <file.json>
```

### 7.3 Session 内部命令

在 TUI 界面中可使用以下斜杠命令：

| 命令 | 功能 |
|------|------|
| `/handoff` | 生成会话上下文摘要，便于新会话继续 |
| `/refactor` | 智能重构（LSP + AST-grep + TDD） |
| `/review-work` | 实现完成后启动多 Agent 评审 |
| `/ai-slop-remover` | 清除 AI 生成的代码异味 |
| `/ralph-loop` | 自指开发循环，持续到完成 |
| `/ulw-loop` | 超轻量工作循环 |
| `/hyperplan` | 对抗式多 Agent 规划 |
| `/caveman` | 切换到极简模式 |
| `/start-work` | 从 Prometheus 计划启动工作 |
| `/stop-continuation` | 停止所有循环机制 |
| `/remove-ai-slops` | 清理分支中的 AI 代码异味 |

---

## 8. MCP 与插件

### 8.1 MCP (Model Context Protocol)

MCP 是 AI 模型与外部工具/数据的标准协议。OpenCode 支持 MCP 服务器：

```powershell
# 管理 MCP
opencode mcp

# 使用 MCP 工具（在 Skill 中）
skill_mcp(
  mcp_name="server-name",
  tool_name="tool-name",
  arguments={...}
)
```

### 8.2 插件系统

```powershell
# 安装一个 npm 包作为插件
opencode plugin <package-name>

# 当前安装的插件
# (见 package.json)
"@opencode-ai/plugin": "1.2.26",
"opencode-antigravity-auth": "^1.6.0"
```

### 8.3 MCP 内建 Skill

在本机环境中，Playwright 集成通过 MCP 提供浏览器自动化能力。

---

## 9. Profile 切换

### 9.1 为什么需要切换？

本机配置了两个 Profile：
1. **jiekou** — 通过 `jiekou.ai` API 代理访问 Anthropic/OpenAI/Gemini 等国外模型
2. **volcengine** — 通过火山引擎访问国内模型（GLM、DeepSeek、Kimi 等）

切换可以在不同网络条件或成本策略下灵活选择。

### 9.2 切换脚本

```powershell
# 执行切换
& "$env:USERPROFILE\.config\opencode\switch-profile.ps1"

# 输出示例：
# Currently active: jiekou
# Backed up: oh-my-openagent.volcengine.json
# Switched to: volcengine
# Restart OpenCode for changes to take effect.
```

### 9.3 工作原理

```
切换前：
  oh-my-openagent.json       ← 当前激活的配置
  oh-my-openagent.jiekou.json       ← jiekou Profile
  oh-my-openagent.volcengine.json    ← volcengine Profile

切换过程：
  1. 检测当前激活的是哪个 Profile
  2. 将当前配置保存到对应备份文件
  3. 将目标 Profile 复制为 oh-my-openagent.json
```

### 9.4 Volcengine Profile 的模型映射

| Agent/类别 | 火山引擎模型 |
|------------|-------------|
| Sisyphus | `go-glm-5`（智谱 GLM-5） |
| Oracle | `minimax-m2.5` |
| Librarian | `deepseek-v3.2` |
| Explore | `go-glm-4.7` |
| visual-engineering | `kimi-k2.5` |
| ultrabrain | `go-glm-5` |
| quick | `minimax-m2.5` |

---

## 10. 最佳实践与工作流

### 10.1 日常工作流

```
1. 启动会话
   opencode

2. 如果要在新目录工作：
   opencode /path/to/project

3. 处理任务时让 AI 自动规划：
   直接描述需求，AI 会创建 Todo 列表

4. 利用子 Agent 并行工作：
   多个独立任务 → 后台并行派发

5. 完成工作后用 review-work 做质量检查
```

### 10.2 任务委托决策树

```
任务来了
  │
  ├─ 简单 / 单文件 → 直接做
  │
  ├─ 前端/UI 工作 → visual-engineering
  │
  ├─ 硬逻辑/算法 → ultrabrain
  │
  ├─ 深度研究+实现 → deep
  │
  ├─ 查资料/文档 → librarian
  │
  ├─ 在代码库中搜索 → explore
  │
  ├─ 架构/调试/卡住了 → oracle
  │
  ├─ 复杂任务先规划 → metis
  │
  └─ 完成后质量检查 → momus / review-work
```

### 10.3 Token 节省技巧

1. **使用 `/caveman` 模式**：减少约 75% Token 消耗
2. **使用 `task_id` 续接**：比重新启动 Agent 节省 70%+ Token
3. **合理选择 variant**：简单任务用 `mini`/`fast`，复杂任务用 `max`
4. **精确描述需求**：模糊的需求会导致多轮对话浪费

### 10.4 高效 Prompt 模板

```typescript
// 标准子 Agent 委托模式
task(
  category="quick",
  load_skills=[],
  prompt="
    TASK: 原子化单一目标
    EXPECTED: 具体产出物
    MUST DO: 必须做的事
    MUST NOT: 禁止做的事  
    CONTEXT: 文件路径、现有模式、约束
  ",
  run_in_background=false
)
```

### 10.5 调试技巧

```powershell
# 查看详细日志
opencode --print-logs --log-level DEBUG

# 诊断问题
opencode debug

# 查看 Token 消耗
opencode stats
```

### 10.6 多会话协作

```powershell
# 终端 1：启动主会话
opencode

# 终端 2（另一个项目目录）：
opencode /another-project

# 或附加到已有服务器
opencode attach <url>
```

---

## 11. 常见问题排查

### 11.1 连接 / API 错误

**症状**: "API connection failed" 或超时

**排查**:
```powershell
# 1. 检查网络
ping api.jiekou.ai

# 2. 检查 API Key 是否有效
opencode auth

# 3. 切换到备用 Profile
& "$env:USERPROFILE\.config\opencode\switch-profile.ps1"

# 4. 或直接换用其他模型
# 在 opencode.json 中修改 "model" 字段
```

### 11.2 模型不可用

**症状**: 某模型报错，自动降级

OhMyOpenCode 的 fallback 机制会自动处理。如果想手动切换：
```powershell
# 查看可用模型
opencode models
```

或编辑 `oh-my-openagent.json` 中对应 Agent 的 `model` 字段。

### 11.3 子 Agent 没按预期工作

```typescript
// 检查是否有 skill 可用
skill(name="appropriate-skill")

// 确保 category 选择正确
// 前端工作 → visual-engineering（不要用 quick）
// 硬逻辑 → ultrabrain（不要用 unspecified-*）
```

### 11.4 会话恢复

```powershell
# 查看历史会话
opencode session list

# 查看特定会话内容
opencode session show <session-id>

# 导出备份
opencode export <session-id>
```

### 11.5 升级后不兼容

```powershell
# 回滚
opencode upgrade <previous-version>

# 或者重装
npm install -g opencode-ai@<version>
```

### 11.6 垃圾文件清理

```powershell
# OpenCode 的临时输出文件位置
# C:\Users\<user>\.local\share\opencode\tool-output\

# 会话数据库位置
# C:\Users\<user>\.local\share\opencode\
```

---

## 附录

### A. 本机目录结构

```
%USERPROFILE%\
├── .config\opencode\                    ← OpenCode 配置目录
│   ├── opencode.json                    ← 主配置（模型/提供商）
│   ├── oh-my-openagent.json             ← OhMyOpenCode 配置（当前激活）
│   ├── oh-my-openagent.jiekou.json      ← jiekou Profile
│   ├── oh-my-openagent.volcengine.json  ← volcengine Profile
│   ├── switch-profile.ps1               ← Profile 切换脚本
│   ├── package.json                     ← 插件依赖
│   ├── skills\                          ← 技能目录
│   │   ├── caveman\
│   │   ├── paper-quick-read\
│   │   ├── pdf-reader\
│   │   ├── planning-with-files-zh\
│   │   ├── ppt-image-first-master\
│   │   └── using-superpowers\
│   └── node_modules\                    ← 插件依赖包
│
└── AppData\Roaming\npm\
    ├── opencode                          ← 启动入口
    └── node_modules\opencode-ai\         ← OpenCode 本体
```

### B. 常用命令速查表

| 操作 | 命令 |
|------|------|
| 启动 TUI | `opencode` |
| 查看版本 | `opencode --version` |
| 查看帮助 | `opencode --help` |
| 管理 API 密钥 | `opencode auth` |
| 列出模型 | `opencode models` |
| 切换 Profile | `& "$env:USERPROFILE\.config\opencode\switch-profile.ps1"` |
| 升级 | `opencode upgrade` |
| 安装插件 | `opencode plugin <package>` |
| 管理会话 | `opencode session <subcommand>` |
| 导出数据 | `opencode export <session-id>` |
| 查看统计 | `opencode stats` |
| 调试模式 | `opencode --print-logs --log-level DEBUG` |

### C. 模型选择参考

| 任务类型 | 推荐 Agent/类别 | 推荐模型 |
|----------|----------------|----------|
| 日常编码 | Sisyphus (主会话) | Claude Opus 4.7 |
| 前端开发 | visual-engineering | Gemini 3.1 Pro |
| 复杂算法 | ultrabrain | GPT-5.5 |
| 代码搜索 | explore | GPT-5.4-mini-fast |
| 文档查阅 | librarian | GPT-5.4-mini-fast |
| 架构评审 | oracle | GPT-5.5 / Opus 4.7 |
| 简单修改 | quick | GPT-5.4-mini |
| 技术写作 | writing | Gemini 3 Flash |
| 图像分析 | multimodal-looker | GPT-5.5 |

---

*生成日期: 2026-05-13*
*基于 OpenCode v1.14.48 · OhMyOpenCode (oh-my-openagent)*
