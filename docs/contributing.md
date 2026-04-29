# 贡献指南

---

## Git 工作流

### 分支命名

```
<type>/<scope>-<简短描述>

# 示例
feat/tool-mcp-client
fix/safety-envelope-audit-log
refactor/skill-registry-hotswap
docs/onboarding-guide
test/brain-replan-edge-cases
```

### 常用 type

| type | 用途 |
|---|---|
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `safety` | 安全相关变更（优先 review） |
| `tool` | tool adapter 相关 |
| `skill` | skill 相关 |
| `embodiment` | 机器人形态适配 |
| `refactor` | 重构（不改行为） |
| `test` | 测试 |
| `docs` | 文档 |
| `chore` | 构建 / 依赖 / CI 等杂项 |

### 开发流程

```bash
# 1. 从 main 拉最新
git checkout main && git pull

# 2. 创建功能分支
git checkout -b feat/tool-mcp-client

# 3. 开发 + 本地测试
uv run pytest tests/unit/
uv run pytest tests/integration/

# 4. 提交（格式见下）
git commit -m "feat(tool): add MCP client stdio transport"

# 5. Push + 开 PR
git push origin feat/tool-mcp-client
```

---

## Commit 规范

格式：`<type>(<scope>): <desc>`

```bash
# 好的示例
feat(tool): add MCP client with stdio transport
fix(safety): prevent SafetyEnvelopeViolation from being swallowed
refactor(skill): extract hotswap logic into SwapHandle
test(brain): cover replan loop exceeded path

# 不好的示例
fix bug                          # 没有 type 和 scope
feat: update stuff               # desc 太模糊
feat(tools/mcp/client.py): xxx  # scope 不用文件路径，用模块名
```

scope 用模块名（`tool` / `skill` / `brain` / `memory` / `critic` / `safety` / `embodiment` / `fleet` / `runtime` / `obs` / `channel` / `config`）。

---

## 编码规范

### 必须遵守

**异步**：模块层全面 `async/await`，业务模块禁止 `asyncio.run()`，事件循环由 `HarnessRuntime` 统一管理。

**类型注解**：公开接口必须完整类型注解，数据类用 Pydantic v2 `BaseModel`，有限枚举用 `Literal` 不用裸 `str`。

**异常**：自定义异常携带 `trace_id` + `robot_id`，**禁止 `try/except Exception: pass` 静默吞错误**，直接抛出由 harness 顶层降级。

**日志**：**禁止 `print()`**，统一通过 `observability.tracer` 结构化记录。

**文件大小**：单文件 ≤ 1000 行，嵌套 ≤ 3 层缩进，超过必须拆模块。

**import 顺序**：stdlib → 第三方 → 本项目，各组空行分隔。

### 注释原则

默认不写注释。只在以下情况写：
- 隐藏约束或隐式不变量
- 绕过特定 bug 的 workaround
- 会让读者意外的行为

不要写解释"代码做了什么"的注释（好的命名本身就是文档），不要引用 issue 号或任务名。

### Tool 实现规范

- 所有 Tool 必须实现 `Tool` Protocol（`tools/base.py`）
- Schema 必须可序列化为 MCP tool definition（兼容 OpenAI function spec）
- 调用必须是 idempotent（除非显式标记 `is_idempotent=False`）
- 必须支持取消（`async def cancel`）
- **禁止**在 ToolAdapter 内部嵌入 retry / circuit breaker / cache 逻辑，这些走 `tools/middleware/`

### Skill 实现规范

- 必须有完整的 `SkillManifest`（`name` / `version` / `embodiment_compat` / `safety_class` / `required_tools` / `data_schema`），缺字段拒绝注册
- Manifest 必须可序列化为 YAML 并通过 Pydantic schema 校验
- 内部**必须**通过 `tools.get(...)` 调用能力，**禁止**直连外部 server

---

## 测试规范

### 必跑路径

提 PR 前本地必须跑通：

```bash
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run ruff check .
uv run ruff format --check .
```

### 测试分类

| 目录 | 说明 | CI |
|---|---|---|
| `tests/unit/` | 纯逻辑，无外部依赖 | 每次 Push 必跑 |
| `tests/integration/` | 含 mock server，自动启动 | 每次 Push 必跑 |
| `tests/hardware/` | 需要真实硬件，标注 `@pytest.mark.hardware` | 默认跳过，手动触发 |

### 涉及 Tool 的测试

必须覆盖以下四种异常路径：
1. schema 违例（输入/输出不符合 schema）
2. backend 不可达（外部 server 宕机）
3. 超时（tool 执行超出 timeout）
4. 取消（调用方主动 cancel）

### SafetyEnvelope

**禁止 mock SafetyEnvelope**。安全测试必须用真实校验逻辑。这是强制要求，不接受例外。

### Skill 端到端测试

Skill 的端到端测试必须运行在真机或高保真 sim 上才能关闭。仅 unit test 通过**不能**声明 skill done。

---

## PR 流程

### 提 PR 前检查

- [ ] 本地测试全部通过（unit + integration）
- [ ] ruff lint + format 无报错
- [ ] 涉及 hardware-bound tool 的 PR 标注 `@pytest.mark.hardware`
- [ ] 新增 tool 覆盖四种异常路径测试
- [ ] 新增 skill 有完整 SkillManifest
- [ ] 没有 `print()` 调试语句遗留

### PR 描述模板

```markdown
## 变更内容
<!-- 一句话说清楚做了什么 -->

## 相关 ADR / Issue
<!-- 例：closes #42，实现 ADR-002 中的 MCP stdio transport -->

## 测试
<!-- 怎么验证这个变更是正确的 -->

## 注意事项（可选）
<!-- reviewer 需要特别关注的地方 -->
```

### Review 要点

**Reviewer 重点看：**
1. SafetyEnvelope 路径有没有被绕过（所有 `embodiment.dispatch` 前必须经过 `safety.check`）
2. Skill 有没有直连外部 server（必须走 `ToolRegistry`）
3. 异常有没有被静默吞掉
4. Tool schema 能否序列化为合法的 MCP definition
5. 新增文件是否放对了目录（框架代码 vs 用户 workspace）

**安全相关 PR**（type=`safety`）需要至少 2 人 approve。

---

## 常见错误

### ❌ 把 retry 逻辑写进 ToolAdapter

```python
# 错误：ToolAdapter 里自己搞 retry
async def invoke(self, args, ctx):
    for i in range(3):
        try:
            return await self._call(args)
        except Exception:
            pass
```

```python
# 正确：adapter 只管调用，retry 走 middleware
async def invoke(self, args, ctx):
    return await self._call(args)
# registry 注册时声明 middleware=[RetryMiddleware(max_retries=3)]
```

---

### ❌ Skill 直连外部 server

```python
# 错误
async def execute(self, subtask, tools, ctx):
    resp = await httpx.get("http://perception-server/detect")
```

```python
# 正确
async def execute(self, subtask, tools, ctx):
    tool = tools.get("perception.detect_objects")
    resp = await tool.invoke({"image": ..., "classes": [...]}, ctx)
```

---

### ❌ try/except 吞掉 SafetyEnvelopeViolation

```python
# 错误：绝对禁止
try:
    await safety.check(cmd, ctx)
except SafetyEnvelopeViolation:
    pass  # 假装没事
```

```python
# 正确：让它抛出，harness 顶层处理停机
await safety.check(cmd, ctx)  # 失败直接抛，不 catch
```

---

### ❌ 用 print 调试

```python
# 错误
print(f"tool called: {tool.name}")

# 正确
tracer.info("tool.invoke", tool_name=tool.name, trace_id=ctx.trace_id)
```
