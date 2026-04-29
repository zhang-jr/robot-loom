# 新成员入门指南

> 欢迎加入 Robot Agent Harness 项目！
> 本文档帮助你在 30 分钟内搭好环境、理解项目结构、提交第一个 PR。
>
> 贡献规范 → `docs/contributing.md`

---

## 项目是什么

**Robot Agent Harness** 是一个模型/形态无关的 LLM/VLM 机器人编排框架。

核心思路：LLM（Brain）通过 tool calling 驱动感知、抓取、记忆、导航等外部能力，harness 负责统一管理 tool 注册、skill 编排、安全校验和多机器人协调。

```
用户指令
   ↓
Channel（CLI / Web / Telegram）
   ↓
Brain（LLM 规划 + tool calling）
   ↓
ToolRegistry → Tool Adapters → 外部能力 Server
   ↓
EmbodimentAdapter → 真实机器人
```

两个核心抽象：
- **Tool**：原子能力（感知、抓取、记忆查询……），强 schema，可幂等重试
- **Skill**：tool 的版本化组合（如 pick&place = detect → grasp → safety_check → dispatch）

---

## 环境搭建

### 前置要求

| 工具 | 版本 | 说明 |
|---|---|---|
| Python | 3.12+ | 严格要求，项目使用新式 async 特性 |
| uv | 最新 | 包管理器，替代 pip/poetry |
| git | 任意 | |

安装 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 克隆与安装

```bash
git clone https://github.com/<org>/robot_harness.git
cd robot_harness

# 创建虚拟环境 + 安装依赖（含 dev extras）
uv sync --extra dev

# 验证安装
uv run python -c "import robot_harness; print('ok')"
```

### 初始化用户 workspace

框架代码与用户配置严格分离（ADR-015）。

```bash
# 创建个人 workspace（存放 robot 配置、自定义 skill 等）
uv run robot-harness init

# workspace 默认路径：~/.robot_harness/workspace/
# 包含：MISSION.md / ROBOT.md / HEARTBEAT.md / tools/ / skills/
```

### 运行测试

```bash
# 所有单元测试（不需要硬件）
uv run pytest tests/unit/

# 集成测试（需要 mock server，自动启动）
uv run pytest tests/integration/

# 硬件测试默认跳过，需要显式开启
uv run pytest tests/hardware/ -m hardware
```

---

## 目录结构速览

```
robot-harness/
├── robot_harness/          # 框架代码（不放用户资产）
│   ├── brain/              # LLM 规划层
│   ├── tools/              # ★ 核心 ★ Tool 抽象 + 各类 adapter
│   │   ├── base.py         # Tool Protocol + ToolRegistry
│   │   ├── middleware/     # retry / cache / circuit breaker / trace
│   │   ├── mcp/            # MCP client + server
│   │   ├── perception/     # 视觉感知 adapter
│   │   ├── grasp/          # 抓取估计 adapter
│   │   ├── memory/         # memory 操作 adapter
│   │   ├── critic/         # 进度评判 adapter
│   │   ├── vla/            # VLA serving adapter
│   │   ├── navigation/     # 导航 adapter
│   │   ├── robot_sdk/      # 机器人本体 SDK adapter
│   │   └── generic/        # shell / fs / web / message
│   ├── skill/              # Skill 版本化注册与编排
│   ├── memory/             # Memory 四类子模块
│   ├── critic/             # Critic + ReplanPolicy
│   ├── embodiment/         # 机器人形态适配层
│   ├── safety/             # SafetyEnvelope（前置，不可绕过）
│   ├── fleet/              # 多机器人协调
│   ├── runtime/            # AgentLoop 主循环
│   ├── channels/           # 用户接入通道
│   ├── observability/      # OTel trace + Prometheus metrics
│   └── errors.py           # 统一异常体系
│
├── workspace_template/     # 用户 workspace 初始化模板
├── examples/               # 可运行的示例
├── tests/
│   ├── unit/               # 纯逻辑单测（CI 必跑）
│   ├── integration/        # 含 mock server（CI 必跑）
│   └── hardware/           # 真机测试（CI 默认跳过）
│
└── docs/                   # 协作文档（本文件所在）
```

---

## 各小组分工入口

| 小组 | 主要负责模块 | 关键文件 | 相关 ADR |
|---|---|---|---|
| **Tool & Skill** | `tools/` + `skill/` | `tools/base.py`, `skill/base.py` | ADR-001, ADR-002, ADR-003, ADR-012 |
| **Brain & Memory** | `brain/` + `memory/` | `brain/base.py`, `memory/base.py` | ADR-004, ADR-005 |
| **Embodiment & Safety** | `embodiment/` + `safety/` | `embodiment/base.py`, `safety/envelope.py` | ADR-007, ADR-008, ADR-009, ADR-016 |
| **Observability & Fleet** | `observability/` + `fleet/` + `runtime/` | `runtime/agent_loop.py` | ADR-006, ADR-010, ADR-013 |

---

## 常见问题

**Q：为什么用 uv 不用 pip/poetry？**
更快，锁文件精确，`uv sync` 一步完成虚拟环境和依赖。

**Q：`robot_harness/` 里为什么没有 VLA / 感知模型的实现？**
设计决策（ADR-001）：重模型推理外置为独立 server，本仓库只实现 client adapter。`tools/<category>/` 里的文件是 adapter，不是模型本身。

**Q：我的 skill / tool 配置应该放哪？**
放 `~/.robot_harness/workspace/`，不要放进 `robot_harness/` 源码目录（ADR-015）。

**Q：能跳过 SafetyEnvelope 测试吗？**
不能。SafetyEnvelope 测试禁止 mock，必须用真实校验逻辑。

**Q：仿真跑通能算 skill 完成吗？**
不算。接触类 skill 必须真机验收（ADR-011）。非接触类 skill（导航/视觉）允许高保真 sim + 至少一次真机抽样。

---

## 需要帮助？

- GitHub Issues：报 bug、提问、讨论设计
- GitHub Discussions：较长的技术讨论
- 找不到任务： 找对应 Phase → 找小组负责人

