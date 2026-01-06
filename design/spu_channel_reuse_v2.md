# SPU Channel Reuse for MPLang v2

**Status**: ✅ Completed & Tested  
**Author**: zhsu  
**Date**: 2026-01-06  
**Related**: [spu_channel_reuse.md](spu_channel_reuse.md), [architecture_v2.md](architecture_v2.md)

## Summary

本设计提案将 v1 中已实现的 SPU channel 复用功能迁移到 MPLang v2 架构，通过 `libspu.link.create_with_channels()` 接口，使 SPU 复用 v2 的 simp worker 通信层，消除额外的 BRPC 端口需求。

## Background

### v1 实现回顾

v1 中已经成功实现了 SPU channel 复用 (2025-12-30 完成)：
- **BaseChannel** (`mplang/v1/runtime/channel.py`): IChannel 适配器，桥接 CommunicatorBase → libspu.link
- **LinkCommunicator** (`mplang/v1/runtime/link_comm.py`): 支持三种模式 (BRPC/Mem/Channels)
- **Simulator** (`mplang/v1/runtime/simulation.py`): ThreadCommunicator 复用
- **Session/Driver** (`mplang/v1/runtime/session.py`, `communicator.py`, `server.py`): HttpCommunicator 复用

**核心成果**：
- ✅ 端口数量减半（6个端口 → 3个端口，3-party 场景）
- ✅ 统一通信栈（HTTP-only，无需 BRPC）
- ✅ 简化部署和配置

### v2 架构差异

v2 采用了全新的架构设计，与 v1 有以下关键差异：

| 层面 | v1 | v2 |
|------|----|----|
| **通信抽象** | `CommunicatorBase` (v1/core/comm.py) | `HttpCommunicator` / `ThreadCommunicator` (v2/backends/simp_worker/) |
| **SPU 管理** | Session 全局状态 | `SPUState` (v2/backends/spu_state.py, DialectState) |
| **运行时** | Simulator / Session | LocalMesh / HttpDriver (simp_worker) |
| **link 创建位置** | Session._seed_spu_env | SPUState.get_or_create |
| **类型系统** | TensorType/TableType only | ScalarType/VectorType/SSType/CustomType 可扩展 |

**关键发现**：
1. v2 的通信层在 `mplang/v2/backends/simp_worker/` 中，不继承自统一的抽象基类
2. v2 的 SPU 管理通过 `SPUState` (DialectState)，在 `spu_impl.py` 调用
3. v2 有两套独立的通信实现：`http.py` (分布式) 和 `mem.py` (本地模拟)

## Motivation

### 为什么需要在 v2 中实现？

1. **v1 即将废弃**：v2 是未来主推版本
2. **架构一致性**：v2 的设计更清晰（dialect state, extensible types），应该享受同样的端口优化
3. **避免重复工作**：用户迁移到 v2 后不应该退步到多端口模式

### 预期收益（与 v1 相同）

- **部署简化**：单端口配置，无需 SPU_PORT_OFFSET 计算
- **资源节省**：减少 50% 端口，统一通信栈
- **开发体验**：统一日志，简化调试

## Design Goals

1. **兼容 v1 实现**：尽量复用 v1 的 BaseChannel 设计，减少重复代码
2. **适配 v2 架构**：遵循 v2 的 DialectState 模式和通信层设计
3. **保持向后兼容**：BRPC 模式作为 fallback，不破坏现有代码
4. **测试覆盖**：与 v1 同等级别的测试覆盖（单元测试 + 集成测试）

## Architecture Analysis

### v2 通信层结构

```
mplang/v2/backends/simp_worker/
├── mem.py          # ThreadCommunicator (本地模拟)
│   └── send(to, key, data)
│   └── recv(frm, key) -> data
├── http.py         # HttpCommunicator (分布式)
│   └── send(to, key, data)
│   └── recv(frm, key) -> data
└── state.py        # SimpWorker (持有 communicator)
```

**关键特点**：
1. **无统一抽象**：ThreadCommunicator 和 HttpCommunicator 是独立实现，无公共基类
2. **相同接口**：两者都有 `send(to, key, data)` 和 `recv(frm, key)` 方法
3. **数据格式**：使用 `serde` 进行 JSON 序列化（安全，但需要支持 bytes）

### v2 SPU 管理流程

```
用户代码
  ↓
mp.device("SPU").jax(fn)
  ↓
spu_impl.py: run_jax_on_spu()
  ↓
SPUState.get_or_create(local_rank, world_size, config, endpoints)
  ↓
如果 endpoints 存在:
  _create_brpc_link()
否则:
  _create_mem_link()
  ↓
libspu.link.create_brpc() / create_mem()
```

### 与 v1 的对比

| 组件 | v1 | v2 |
|------|----|----|
| BaseChannel 位置 | v1/runtime/channel.py | **已新建**: v2/backends/channel.py |
| Communicator 抽象 | CommunicatorBase (统一) | 无基类（duck typing） |
| Link 创建入口 | Session._seed_spu_env | SPUState.get_or_create |
| Link 模式选择 | LinkCommunicator.__init__ | SPUState.get_or_create |
| 测试位置 | tests/v1/runtime/test_channel.py | tests/v2/backends/test_channel.py |

## Proposed Solution

### 1. 新建 BaseChannel (v2 版本)

**文件位置**: `mplang/v2/backends/channel.py` (已实现)

```python
# 伪代码示意
class BaseChannel(libspu.link.IChannel):
    """Bridge v2 communicator to SPU IChannel interface.
    
    Supports both ThreadCommunicator and HttpCommunicator via duck typing.
    """
    
    def __init__(
        self, 
        comm: ThreadCommunicator | HttpCommunicator,  # Duck typing
        local_rank: int,
        peer_rank: int,
        tag_prefix: str = "spu",
    ):
        # 与 v1 基本相同，但适配 v2 的 communicator 接口
        pass
```

**与 v1 的差异**：
1. **无类型约束**：v2 的 communicator 没有统一基类，使用 duck typing
2. **serde 支持**：v2 使用 JSON 序列化，需要处理 bytes 编码（base64）
3. **导入路径**：适配 v2 的模块结构

### 2. 修改 SPUState

**文件位置**: `mplang/v2/backends/spu_state.py`

```python
class SPUState(DialectState):
    def get_or_create(
        self,
        local_rank: int,
        spu_world_size: int,
        config: spu.SPUConfig,
        spu_endpoints: list[str] | None = None,
        # 新增参数
        communicator: ThreadCommunicator | HttpCommunicator | None = None,
        parties: list[int] | None = None,  # SPU parties 的全局 rank 列表
    ) -> tuple[spu_api.Runtime, spu_api.Io]:
        # 如果提供了 communicator，使用 Channels 模式
        if communicator is not None:
            link = self._create_channels_link(
                local_rank, spu_world_size, communicator, parties
            )
        elif spu_endpoints:
            link = self._create_brpc_link(local_rank, spu_endpoints)
        else:
            link = self._create_mem_link(local_rank, spu_world_size)
        # ...
    
    def _create_channels_link(
        self,
        local_rank: int,        # SPU local rank (已转换)
        spu_world_size: int,    # SPU world size
        communicator,           # Worker communicator
        parties: list[int],     # SPU parties 全局 ranks
    ) -> libspu.link.Context:
        """Create link using custom channels (NEW).
        
        Note: local_rank and parties conversion is already done by exec_impl.
        parties[local_rank] == global_rank of this worker.
        """
        from mplang.v2.backends.channel import BaseChannel
        
        # 创建 channels 列表
        # parties 已经是按 SPU local rank 排序的全局 rank 列表
        global_rank = parties[local_rank]
        
        channels = []
        for idx, peer_global_rank in enumerate(parties):
            if idx == local_rank:  # 使用 local_rank 判断自己
                channel = None  # Self channel
            else:
                # 创建到其他 SPU party 的 channel
                channel = BaseChannel(communicator, global_rank, peer_global_rank)
            channels.append(channel)
        
        # 创建 descriptor
        desc = libspu.link.Desc()
        desc.recv_timeout_ms = 100 * 1000
        for idx in range(spu_world_size):
            desc.add_party(f"P{idx}", f"dummy_{parties[idx]}")
        
        return libspu.link.create_with_channels(desc, local_rank, channels)
```

### 3. 修改 spu_impl.py

**文件位置**: `mplang/v2/backends/spu_impl.py`

在 `exec_impl` 中传递 communicator 和 parties 到 SPUState：

```python
@spu.exec_p.def_impl
def exec_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    # ... (现有的 rank 转换逻辑不变)
    
    parties = context.current_parties  # 已有
    global_rank = context.rank         # 已有
    local_rank = parties.index(global_rank)  # 已有
    spu_world_size = len(parties)      # 已有
    
    # 获取当前 worker 的 communicator (NEW)
    communicator = context.communicator  # SimpWorker 已有此属性
    
    # 传递给 SPUState (修改)
    runtime, io = spu_state.get_or_create(
        local_rank, 
        spu_world_size, 
        config, 
        spu_endpoints,
        communicator=communicator,  # NEW
        parties=parties,            # NEW (list of global ranks)
    )
    # ... (其余逻辑不变)
```

**关键发现** (基于代码审查):
- ✅ **Rank 映射已完成**: `exec_impl` 已经完成 `global_rank -> local_rank` 转换
- ✅ **Parties 已获取**: `context.current_parties` 就是 SPU 参与的全局 ranks 列表
- ✅ **无需额外 mask**: parties 本身就是有序的 SPU ranks，无需单独传 mask

### 4. HttpCommunicator 增强 (支持 bytes)

**文件位置**: `mplang/v2/backends/simp_worker/http.py`

```python
class HttpCommunicator:
    def send(self, to: int, key: str, data: Any) -> None:
        # 检测 SPU channel (tag prefix "spu:")
        if key.startswith("spu:") and isinstance(data, bytes):
            # 直接 base64 编码 bytes
            payload = base64.b64encode(data).decode('ascii')
            is_raw_bytes = True
        else:
            # 正常 serde 序列化
            payload = serde.dumps_b64(data)
            is_raw_bytes = False
        
        resp = self.client.put(url, json={
            "data": payload,
            "from_rank": self.rank,
            "is_raw_bytes": is_raw_bytes  # NEW
        })
    
    def recv(self, frm: int, key: str) -> Any:
        data = self._mailbox.pop(key)
        # 如果是 raw bytes，直接返回
        if isinstance(data, bytes):
            return data
        # 否则正常 serde 反序列化
        return data
```

**Server 端修改** (需要找到对应的 FastAPI 端点处理函数)。

### 5. ThreadCommunicator 支持 (已经支持)

`ThreadCommunicator` 直接传递对象，天然支持 bytes，无需修改。

## Implementation Plan

### ✅ Phase 1: Core Infrastructure (Completed)

- [x] 创建 `mplang/v2/backends/channel.py`
  - [x] 实现 BaseChannel (复用 v1 逻辑，适配 duck typing)
  - [x] 单元测试：`tests/v2/backends/test_channel.py` (9 tests passed)

### ✅ Phase 2: SPUState Integration (Completed)

- [x] 修改 `SPUState.get_or_create` 支持 communicator 参数
- [x] 实现 `_create_channels_link` 方法
- [x] 添加 cache key 包含 link_mode ("channels")

### ✅ Phase 3: Worker Integration (Completed)

- [x] 修改 `spu_impl.py` 传递 communicator 和 parties
- [x] ThreadCommunicator: 修复 mailbox 机制（dict → deque）
- [x] HttpCommunicator: 增强 bytes 支持（tag prefix + is_raw_bytes）
- [x] CommRequest: 添加 is_raw_bytes 字段
- [x] /comm/{key} 端点: 处理 raw bytes

### ✅ Phase 4: Testing & Validation (Completed)

- [x] 单元测试：`tests/v2/backends/test_channel.py` (9 tests)
- [x] 集成测试：LocalMesh (test_spu_channels_mode_simulation passed)
- [x] 验证并行创建：无需额外 threading（LocalMesh 自带并行）
- [x] 修复 ThreadCommunicator mailbox overflow 问题

## Key Differences from v1

| 方面 | v1 | v2 |
|------|----|----|
| **BaseChannel 导入** | `from mplang.v1.runtime.channel` | `from mplang.v2.backends.channel` |
| **Communicator 类型** | `CommunicatorBase` (抽象基类) | Duck typing (ThreadCommunicator \| HttpCommunicator) |
| **Link 创建入口** | `LinkCommunicator.__init__` | `SPUState._create_channels_link` |
| **测试文件** | `tests/v1/runtime/test_channel.py` | `tests/v2/backends/test_channel.py` |
| **Worker 状态** | Session (全局) | SimpWorker (per-interpreter) |

## Open Questions

### ✅ 已解决

1. **Mask 传递** ✅
   - **结论**: 不需要单独传递 mask
   - **原因**: `exec_impl` 中的 `context.current_parties` 已经是 SPU 参与的全局 ranks 列表
   - **实现**: 直接使用 `parties` 参数即可

2. **Cache Key 设计** ✅
   - **结论**: 保持现有设计，添加 link_mode 区分
   - **原因**: 快速迭代中，无需考虑向后兼容

3. **向后兼容** ✅
   - **结论**: 无需考虑
   - **原因**: 没有现有用户，快速迭代中

### ✅ 已验证

4. **并行创建** ✅
   - **LocalMesh**: 不需要额外 threading（多线程 worker 自带并行）
   - **HttpDriver**: 不需要额外 threading（多进程 worker 天然并行）
   - **TestSend/TestRecv**: 恢复正常握手逻辑（mailbox 修复后可以正常工作）

## Success Criteria

- [x] v2 的 SPU 可以复用 ThreadCommunicator (LocalMesh)
- [x] v2 的 SPU 可以复用 HttpCommunicator (HttpDriver)
- [x] 单元测试覆盖：9 个 BaseChannel 单元测试
- [x] 集成测试通过：LocalMesh (3-party) + HttpDriver (2-party)
- [x] 关键修复：ThreadCommunicator mailbox 使用 (from_rank, tag) 作为 key
- [x] 文档更新（设计文档 + 实现总结）

## Key Implementation Insights

### 1. ThreadCommunicator Mailbox Fix（核心修复）

**问题根源**: v2 原始的 `ThreadCommunicator._mailbox` 只使用 `tag` 作为 key，忽略了 `recv(frm, key)` 的 `frm` 参数。这导致：
- 多个 peer 向同一个 receiver 发送相同 tag 时，消息会混淆
- 无法区分是哪个 peer 发送的消息
- SPU 的并发通信（如 ALLGATHER）会导致 "Mailbox overflow" 错误

**正确的修复**：Mailbox 使用 `(from_rank, tag)` 作为复合 key：
```python
# Before (错误): 只用 tag
self._mailbox: dict[str, Any] = {}
# 问题：收到 peer 0 和 peer 2 的相同 tag 会冲突

# After (正确): 用 (from_rank, tag)
self._mailbox: defaultdict[tuple[int, str], deque[Any]] = defaultdict(deque)
#                              ↑         ↑        ↑
#                          from_rank   tag    队列(支持同一sender多次发送)

def recv(self, frm: int, key: str) -> Any:
    mailbox_key = (frm, key)  # 使用 frm 参数！
    return self._mailbox[mailbox_key].popleft()

def _on_receive(self, frm: int, key: str, data: Any) -> None:
    mailbox_key = (frm, key)  # 区分不同发送方
    self._mailbox[mailbox_key].append(data)
```

**为什么需要两层**：
1. **第一层 (from_rank, tag)**：区分不同发送方的相同 tag
2. **第二层 deque**：支持同一发送方多次发送相同 tag（队列化）

### 2. TestSend/TestRecv 握手逻辑

**之前的错误理解**: 以为 mailbox overflow 是因为握手冲突，所以改成 no-op。

**现在的正确实现**: Mailbox 修复后，TestSend/TestRecv 可以正常工作：
```python
def TestSend(self, timeout: int) -> None:
    test_data = b"\x00"  # 1-byte handshake
    self.Send("__test__", test_data)

def TestRecv(self) -> None:
    test_data = self.Recv("__test__")
    if test_data != b"\x00":
        logging.warning(f"Unexpected handshake: {test_data!r}")
```

握手逻辑现在完全正常，因为 mailbox 使用 `(peer_rank, "spu:__test__")` 作为 key，不会冲突。

### 3. HttpCommunicator Bytes Handling

SPU 发送 raw bytes，需要区分于 simp 的 serde 序列化：
- **检测**: `key.startswith("spu:") and isinstance(data, bytes)`
- **编码**: `base64.b64encode(data).decode()` + `is_raw_bytes=True`
- **解码**: 服务端根据 `is_raw_bytes` 字段条件解码

## References

- v1 实现: `design/spu_channel_reuse.md`
- v1 BaseChannel: `mplang/v1/runtime/channel.py`
- v2 SPUState: `mplang/v2/backends/spu_state.py`
- v2 Communicators: `mplang/v2/backends/simp_worker/{http,mem}.py`
- libspu API: `spu.libspu.link.{IChannel,create_with_channels}`

---

## Status: ✅ DONE

All 4 phases complete. SPU Channels mode fully functional in v2:
- **9 BaseChannel unit tests** passing (`tests/v2/backends/test_channel.py`)
- **LocalMesh integration** passing (`test_spu_channels_mode_simulation` - 3 parties)
- **HttpDriver integration** passing (`test_spu_computation` - 2 parties, high-level device API)
- **ThreadCommunicator** mailbox 正确实现：使用 `(from_rank, tag)` 复合 key + deque
- **BaseChannel** TestSend/TestRecv 恢复正常握手逻辑
- **HttpCommunicator** enhanced with raw bytes support (base64 + is_raw_bytes flag)

### Test Summary

```bash
# Unit tests (9 tests)
uv run pytest tests/v2/backends/test_channel.py -v
# ✅ 9 passed

# Integration tests
uv run pytest tests/v2/backends/test_spu_impl.py -v
# ✅ test_spu_e2e_simulation (原有测试，未破坏)
# ✅ test_spu_channels_mode_simulation (新增，3-party LocalMesh)

uv run pytest tests/v2/backends/simp_driver/test_http.py::TestDriverExecution::test_spu_computation -v
# ✅ test_spu_computation (高层 device API，2-party HttpDriver)
```

### 核心文件

1. **mplang/v2/backends/channel.py** (新建, 224 行)
2. **mplang/v2/backends/spu_state.py** (修改, +30 行)
3. **mplang/v2/backends/spu_impl.py** (修改, +3 行)
4. **mplang/v2/backends/simp_worker/http.py** (修改, +15 行)
5. **mplang/v2/backends/simp_worker/mem.py** (修改, mailbox 队列化)

**准备合并！🚀**
