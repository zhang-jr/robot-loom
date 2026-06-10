# 架构概览

---

## 一句话定位

**Robot Agent Harness** 是 LLM/VLM Brain 和各类外部能力 Server 之间的编排层。

Brain 不直接控制机器人，**Brain 通过 tool calling 驱动一切**。harness 负责：tool 注册与路由、skill 版本化编排、安全前置校验、多机器人协调、全链路 trace。

---

## 概念分层

| 层 | 职责 | 频率 / 部署 |
|---|---|---|
| **Channel** | 用户接入：CLI / Web / Telegram / Discord / 飞书 | 事件驱动 |
| **Brain** | LLM/VLM 慢思考 + tool calling 决策 | 1–7 Hz，可云端 |
| **ToolRegistry** | tool 注册、schema 校验、路由、调用追踪 | 同步 / 异步混合 |
| **SkillRegistry** | skill = tool 组合的版本化包装 | 中频 |
| **Memory** | 持久记忆（object / place / episodic / semantic） | 异步 |
| **Critic** | 进度评判（supervisor，不是 controller） | 1–3 Hz |
| **Safety** | 前置校验，违反就拒绝 | 同步，必须 < 5ms |
| **ToolAdapter** | 对接外部 server 的 client | 因 tool 而异 |
| **EmbodimentAdapter** | 对接 robot 本体 SDK | 50–200 Hz |
| **External Servers** | 感知 / 抓取 / 记忆 / critic / VLA / 导航服务 | **不在本仓库** |

---

## 架构图

```
Application
   │
   ▼
┌────────────────────────────────────────────────────┐
│ Channel  (CLI / Web / Telegram / Discord / Feishu) │
└────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────┐
│ Harness Core                                       │
│   ├─ AgentLoop  (Brain ⇄ tool calling 主循环)      │
│   ├─ ToolRegistry      ←─── 一等抽象                │
│   ├─ SkillRegistry     (tool 组合 + manifest 版本) │
│   ├─ Memory            (跨 session 持久化)          │
│   ├─ Critic            (进度评判)                   │
│   ├─ SafetyEnvelope    (前置校验)                   │
│   └─ Scheduler         (heartbeat / cron)          │
└────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────┐
│ Tool Adapters  (本仓库实现 client，server 外置)     │
│   ├─ perception/   → slow-loop grounding 感知 only │
│   │                  (mid-loop 反应式感知下沉本体)  │
│   ├─ grasp/        → slow-loop hint generator      │
│   │                  (闭环抓取已下沉到本体动词)     │
│   ├─ memory/       → SpatialMemory-like server     │
│   ├─ critic/       → VLAC-like progress supervisor │
│   ├─ vla/          → VLA serving (chunked, 非 tick)│
│   ├─ navigation/   → slow-loop path planning only  │
│   ├─ robot_sdk/    → per-robot agent_server        │
│   │                  暴露本体动词 (reactive_grasp / │
│   │                  visual_servo_to / move_to_pose)│
│   └─ generic/      → shell / fs / web / message    │
└────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────┐
│ External Servers / Hardware  (独立部署)             │
│   GPU 推理服务  ·  本体硬件  ·  仿真器               │
└────────────────────────────────────────────────────┘
```

---

## 两个核心抽象：Tool vs Skill

这是最容易混淆的地方，务必区分清楚（ADR-003）。

### Tool — 原子能力

```
特征：无状态或局部状态 / 强 schema / 可幂等重试 / 可取消
示例：
  perception.detect_objects(image, classes) → detections
  grasp.estimate_pose(image, target_id)     → grasp_pose
  robot_sdk.move_joint(robot_id, target)    → handle
  memory.query(query, top_k)               → hits
```

Tool 是 LLM 自由组合的乐高积木。

### Skill — tool 的版本化组合

```
特征：有 manifest（name/version/embodiment_compat/safety_class/required_tools）
      可 hotswap / rollback / 按形态路由
示例：
  pick_object v1.2.0 = detect → grasp_estimate → safety_check → move → critic_judge
```

Skill 是"已验证安全、可版本治理"的组合。

**判断原则**：这个能力需要版本化治理、形态适配、安全分级吗？是 → Skill。只是一个原子操作吗？是 → Tool。

---

## 数据流：一次 pick&place 的完整路径

下面的流程符合 ADR-019 的**三层闭环边界**：mid/tight-loop 反应式闭环（视觉伺服、
grasp 微调、力控反射）完全跑在本体 agent_server 内部，harness 只下高层意图、
等动词级 `CompletionVerdict`，再由 supervisor critic 做任务级验收。

```
用户: "把桌上的杯子放到托盘上"
   │
   ▼ Channel.receive()
   │
   ▼ Brain.decide(task, memory_view, tools)
   │   └─ 查询 memory：桌子在哪？杯子上次在哪？
   │   └─ 输出：调用 pick_object skill
   │
   ▼ SkillRegistry.get("pick_object")
   │
   ▼ Skill.execute()  — declarative pick (harness 内)
   │   ├─ tools.get("perception.ground_phrase").invoke({phrase:"the mug"})
   │   │   → 产生 target_hint  (slow-loop B 类感知，0.5–7 Hz)
   │   │
   │   ├─ SafetyEnvelope.check(高层意图: target_pose + constraints)
   │   │   → 校验可达性 / 工作空间冲突；紧停反射由本体兜底（ADR-019）
   │   │
   │   ├─ tools.get("robot_sdk.reactive_grasp").invoke(
   │   │       {target_hint, constraints})
   │   │   → 本体动词 dispatch → 等 CompletionVerdict
   │   │   ┌─────────────────────────────────────────────────┐
   │   │   │ 本体 agent_server 内 (harness 看不见):           │
   │   │   │   30 Hz 视觉伺服 + grasp 在线微调 +              │
   │   │   │   力/IMU 融合 + 紧停反射                          │
   │   │   └─────────────────────────────────────────────────┘
   │   │   → 返回 CompletionVerdict(success/partial/failed,
   │   │                            robot_state_snapshot)
   │   │
   │   └─ tools.get("critic.judge").invoke(frame)
   │       → 外部 supervisor 视角，看视频帧判定任务级进度
   │         (与本体 CompletionVerdict 形成对偶，ADR-010 / ADR-019)
   │
   ▼ ReplanPolicy 决策（本体 verdict + critic verdict + sensor 联合）
   │   ├─ 双 verdict 一致 → 接受
   │   ├─ 冲突           → CriticDisagreementError → 追加 critic-feedback 消息重规划
   │   ├─ completion     → 任务完成，写 EpisodicMemory
   │   ├─ failure        → 追加 critic-feedback 消息，下一 turn 重规划（ADR-025）
   │   └─ unchanged      → 超时计数，触发降级
   │
   ▼ trace 写入 OTel，全链路可观测（含 cognitive_scaffold_trail）
```

**频率分层（ADR-019）**：

| 闭环 | 频率 | 物理位置 | 在本流程中 |
|---|---|---|---|
| Tight | 100–1000 Hz | on-robot | 关节伺服、紧停反射 —— 本体内 |
| Mid | 5–30 Hz | **on-robot** | 视觉伺服、grasp 微调 —— `reactive_grasp` 内部 |
| Slow | 0.5–7 Hz | harness | Brain decide / skill 编排 / critic / replan |

---

## 关键设计约束

以下是新成员最容易违反的约束，每条都有明确理由：

**① Server 外置，本仓库只有 client adapter**

`tools/<category>/` 里的文件是 adapter，不实现模型推理。判定：需要加载 > 100MB 权重 / 需要 GPU 长驻 → 必须是外部 server。（ADR-001）

**② MCP 是默认 tool 接入协议**

新接入的能力优先包装为 MCP server。只有协议确实不适配（超高频 / 硬件直连）时才写 native adapter，且必须写 ADR 记录原因。（ADR-002）

**③ Brain 与 Tool 频率解耦**

Brain 1–7 Hz，本体控制 50–200 Hz。Brain 的 tool call 是异步 fire-and-await，await 的是关键完成事件，不是每个 tick。禁止跨层共享内部状态。（ADR-004）

**④ SafetyEnvelope 前置不可绕过**

所有 `EmbodimentAdapter.dispatch` 前必须经过 `SafetyEnvelope.check()`。失败抛 `SafetyEnvelopeViolation`，禁止 catch 后放行。测试中禁止 mock SafetyEnvelope。（ADR-007）

**⑤ Memory 禁止用 prompt 拼接模拟**

Brain 通过 `memory.query` tool 主动检索，结果作为 grounding evidence 注入 prompt。禁止把历史记录塞进 system prompt。（ADR-005）

**⑥ 单机器人是 fleet_size=1 的特例**

所有接口从 Phase 1 起都带 `robot_id`，不为单机做专门优化。（ADR-008）

---

## 核心接口速查

```python
# Tool：原子能力
class Tool(Protocol):
    name: str
    schema: ToolSchema
    backend: ToolBackend          # 'mcp' | 'http' | 'native' | 'inproc'
    async def invoke(self, args: dict, ctx: ToolContext) -> ToolResult: ...
    async def cancel(self, ctx: ToolContext) -> None: ...

# Skill：版本化 tool 组合
class Skill(Protocol):
    manifest: SkillManifest       # name / version / embodiment_compat / safety_class / ...
    async def execute(self, subtask, tools: ToolRegistry, ctx) -> SkillResult: ...
    async def rollback(self, ctx) -> None: ...

# Brain：LLM 决策层（原生 tool-use 对话由 AgentLoop 拥有并增长，ADR-025；
# replan = loop 往同一对话追加 critic-feedback 消息，无独立入口）
class Brain(Protocol):
    async def decide(
        self, messages, tools, *, trace_id="", robot_id=""
    ) -> BrainDecision: ...

# SafetyEnvelope：前置校验
class SafetyEnvelope(Protocol):
    async def check(self, cmd: EmbodimentCommand, ctx) -> SafetyVerdict:
        # 失败必须抛 SafetyEnvelopeViolation，不允许返回 False 静默放行
```

---

## 异常体系结构

```
HarnessError
├── BrainError          （规划层：超时 / 输出非法 / 重规划死循环）
├── ToolError           （tool 层：找不到 / server 不可达 / schema 违例 / 超时 / 取消）
├── SkillError          （skill 层：找不到 / 版本不匹配 / manifest 非法 / 安全分级冲突）
├── MemoryError         （内存服务：不可用 / 冲突 / 查询为空）
├── CriticError         （评判层：服务宕机 / 与传感器严重冲突）
├── EmbodimentError     （硬件层：未就绪 / robot 离线 / dispatch 超时）
├── SafetyError
│   └── SafetyEnvelopeViolation   ← 不可恢复，必须停机 + 持久化审计
└── FleetError          （多机器人：容量超限 / 租约冲突 / 形态不匹配）
```

所有异常携带：`trace_id` + `robot_id` + `subtask_id` + `tool_name`（如适用）。

---

## 技术选型速查

| 组件 | 方案 |
|---|---|
| 语言 | Python 3.12+ async |
| Tool 协议（默认） | MCP（stdio + Streamable HTTP） |
| Brain LLM 后端 | LiteLLM（OpenAI / Anthropic / vLLM / Ollama） |
| Skill manifest | YAML + Pydantic v2 |
| 序列化 | msgpack（高频）+ JSON（配置/低频） |
| Trace | OpenTelemetry → Langfuse / Jaeger |
| 指标 | Prometheus |
| 测试 | pytest + pytest-asyncio |
| 包管理 | uv + hatchling |
| Robot SDK 协议 | HTTP/WS（per-robot FastAPI agent_server） |
