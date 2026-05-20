# 编码风格指南

> 本文档是编码规范的协作者版本，加了具体示例方便对照执行。
> 贡献流程（分支、commit、PR）→ `docs/contributing.md`

---

## 文件与模块组织

### 文件大小与嵌套

- 单文件 **≤ 1000 行**，超过必须拆模块
- 嵌套 **≤ 3 层缩进**，超过必须提函数

```python
# ❌ 嵌套过深
async def execute(self, subtask, tools, ctx):
    if condition_a:
        for item in items:
            if condition_b:
                async with lock:
                    result = await do_something()  # 第 5 层

# ✅ 提函数压平
async def execute(self, subtask, tools, ctx):
    if condition_a:
        await self._process_items(items, ctx)

async def _process_items(self, items, ctx):
    for item in items:
        if condition_b:
            await self._acquire_and_run(item, ctx)
```

### 目录 mirrors 逻辑架构

物理目录结构要让人一眼看出模块职责。新增模块时，放在与职责匹配的目录下，不要因为"先随便放"而堆进 `utils/` 或根目录。

```
tools/perception/mcp_bundle.py      ✅ 职责清晰
tools/utils/perception.py           ❌ utils 是垃圾桶
```

### 框架代码 vs 用户资产

| 类型 | 放在 |
|---|---|
| 框架逻辑 | `robot_harness/` |
| 用户自定义 skill / tool | `~/.robot-loom/workspace/skills/` |
| Robot 配置 / 私有 prompt | `~/.robot-loom/workspace/` |

**禁止**把用户资产放进 `robot_harness/` 源码目录。

---

## Import 顺序

三段式，各段之间空行分隔：

```python
# 1. stdlib
import asyncio
from typing import Literal

# 2. 第三方
import httpx
from pydantic import BaseModel

# 3. 本项目
from robot_harness.tools.base import Tool, ToolContext, ToolResult
from robot_harness.errors import ToolTimeoutError
```

---

## 类型注解

### 公开接口必须完整注解

```python
# ❌ 没有注解
async def invoke(self, args, ctx):
    ...

# ✅ 完整注解
async def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
    ...
```

### 数据类用 Pydantic v2

```python
from pydantic import BaseModel

class ToolResult(BaseModel):
    success: bool
    output: dict
    latency_ms: float
    trace_id: str
```

### 有限枚举用 Literal，不用裸 str

```python
# ❌
backend: str  # 只能靠注释说明合法值

# ✅
from typing import Literal
backend: Literal["mcp", "http", "native", "inproc"]
```

---

## 异步规范

### 全面 async/await

模块层所有 I/O 操作都用 async。

```python
# ❌ 同步阻塞
def invoke(self, args, ctx):
    return requests.post(url, json=args).json()

# ✅ 异步非阻塞
async def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=args)
        return ToolResult(**resp.json())
```

### 禁止业务模块 asyncio.run()

事件循环由 `HarnessRuntime` 统一管理。业务代码里出现 `asyncio.run()` 是错误。

```python
# ❌ 业务模块里
asyncio.run(self.invoke(args, ctx))

# ✅ 正常 await
result = await self.invoke(args, ctx)
```

---

## 异常处理

### 禁止静默吞错误

```python
# ❌ 吞掉异常，调试地狱
try:
    result = await tool.invoke(args, ctx)
except Exception:
    pass

# ❌ 吞掉再返回 None，调用方不知道出了什么问题
try:
    result = await tool.invoke(args, ctx)
except Exception:
    return None

# ✅ 只 catch 你真的要处理的异常，其余让它抛
try:
    result = await tool.invoke(args, ctx)
except ToolTimeoutError:
    await ctx.cancel()
    raise  # 或者包装成上层异常再 raise
```

### 自定义异常携带上下文

```python
# ❌ 没有上下文，出问题不知道在哪
raise ToolTimeoutError("timeout")

# ✅ 携带 trace_id / robot_id / tool_name
raise ToolTimeoutError(
    message="timeout after 5s",
    trace_id=ctx.trace_id,
    robot_id=ctx.robot_id,
    tool_name=self.name,
)
```

### SafetyEnvelopeViolation 绝不 catch 后放行

```python
# ❌ 绝对禁止
try:
    await safety.check(cmd, ctx)
except SafetyEnvelopeViolation:
    pass  # 机器人会出事故

# ✅ 让它抛，harness 顶层触发紧急停机
await safety.check(cmd, ctx)
```

---

## 日志与可观测性

### 禁止 print()

```python
# ❌
print(f"invoking tool: {self.name}")
print(f"result: {result}")

# ✅
from robot_harness.observability.tracer import tracer

tracer.info(
    "tool.invoke.start",
    tool_name=self.name,
    trace_id=ctx.trace_id,
    robot_id=ctx.robot_id,
)
```

调试时也不用 print，用 `tracer.debug()`，上生产时日志级别控制。

---

## 注释原则

**默认不写注释**。代码命名本身就是文档。

只在以下情况写注释：

```python
# ✅ 隐藏约束：解释非显而易见的原因
# MCP stdio transport 不支持并发调用，必须串行化
async with self._lock:
    result = await self._send(request)

# ✅ 绕过特定 bug 的 workaround
# httpx 在 Windows 上 ProxyTransport 有内存泄漏（httpx#2341），
# 每次请求创建新 client 而不是复用
async with httpx.AsyncClient() as client:
    ...

# ✅ 会让读者意外的行为
# timeout=0 表示不限时，不是立即超时
await tool.invoke(args, ctx.with_timeout(0))
```

不要写的注释：

```python
# ❌ 解释"做了什么"——好的命名已经说清楚了
# 调用工具
result = await tool.invoke(args, ctx)

# ❌ 引用任务号（任务号会过期）
# 为了修复 issue #123 添加的逻辑
if edge_case:
    ...

# ❌ 多余的文档字符串
async def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
    """
    调用 tool。
    参数：args 是参数字典，ctx 是上下文。
    返回：ToolResult。
    """
```

---

## 命名规范

### 模块与文件

```
tools/perception/mcp_bundle.py      # 小写 + 下划线
skill/builtin/pick.py
embodiment/arm/generic_6dof.py
```

### 类名

```python
class YoloAdapter:          # ✅ PascalCase
class MCPClient:            # ✅ 缩写全大写
class toolAdapter:          # ❌
```

### 函数 / 变量

```python
async def get_camera_frame(camera: str) -> Frame:   # ✅ snake_case
robot_id = "arm_01"                                  # ✅
robotId = "arm_01"                                   # ❌ camelCase
```

### 常量

```python
MAX_RETRY_COUNT = 3         # ✅ 大写 + 下划线
DEFAULT_TIMEOUT_S = 5.0
```

---

## 不要做的事（Anti-patterns 速查）

最常见的：

| ❌ 禁止 | ✅ 正确 |
|---|---|
| `print()` 调试 | `tracer.debug()` |
| `try/except Exception: pass` | 只 catch 要处理的，其余抛出 |
| Tool adapter 内嵌 retry / cache | 走 `tools/middleware/` |
| Skill 直连外部 server | 必须通过 `tools.get(...)` |
| catch `SafetyEnvelopeViolation` 后继续 | 让它抛，顶层停机 |
| 业务模块 `asyncio.run()` | 正常 `await` |
| 用 prompt 拼接模拟 memory | `memory.query()` tool |
| 单文件 > 1000 行 | 拆模块 |
| 嵌套 > 3 层 | 提函数 |
| 裸 `str` 表示有限枚举 | `Literal[...]` |
