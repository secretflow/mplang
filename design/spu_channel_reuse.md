# SPU Channel Reuse Design

**Status**: Draft  
**Author**: zhsu  
**Date**: 2025-12-29  
**Related**: [architecture_v2.md](architecture_v2.md)

## Summary

本设计提案通过 `libspu.link.create_with_channels()` 新接口，实现 SPU 复用 MPLang 通信层，消除额外的 BRPC 端口需求，简化部署和配置。

## Motivation

### 当前问题

MPLang v1 中 SPU 使用独立的 BRPC 通信层，导致：

1. **端口冗余**：每个节点需要两个端口
   - MPLang HTTP 端口（控制层）：如 `8100-8102`
   - SPU BRPC 端口（数据层）：如 `8200-8202`（偏移 +100 或 +1000）

2. **配置复杂**：需要额外的端口计算和管理
   ```python
   # session.py:188
   SPU_PORT_OFFSET = 100
   new_addr = f"{parsed.hostname}:{parsed.port + SPU_PORT_OFFSET}"
   ```

3. **资源浪费**：两套独立的通信栈（HTTP + BRPC）

4. **调试困难**：通信路径分散，难以统一追踪

### SPU 新特性

SPU (libspu) 新版本通过 pybind11 提供了 `create_with_channels` 接口：

```cpp
// C++ 侧
m.def("create_with_channels",
  [](const ContextDesc& desc, size_t self_rank,
     std::vector<std::shared_ptr<IChannel>> channels) {
    py::gil_scoped_release release;
    auto ctx = std::make_shared<yacl::link::Context>(
        desc, self_rank, std::move(channels), nullptr, false);
    ctx->ConnectToMesh();
    return ctx;
  },
  py::arg("desc"), py::arg("self_rank"), py::arg("channels"));
```

```python
# Python 侧（验证通过）
libspu.link.create_with_channels(
    desc: libspu.link.Desc,
    self_rank: int,
    channels: Sequence[libspu.link.IChannel]
) -> libspu.link.Context
```

**IChannel 接口**：

```python
class IChannel:
    def send(self, tag: str, data: bytes) -> None: ...
    def recv(self, tag: str) -> bytes: ...
    def send_async(self, tag: str, data: bytes) -> None: ...
    def send_async_throttled(self, tag: str, data: bytes) -> None: ...
    def test_send(self, tag: str) -> bool: ...
    def test_recv(self, tag: str) -> bool: ...
    def wait_link_task_finish(self) -> None: ...
    def abort(self) -> None: ...
    def set_throttle_window_size(self, size: int) -> None: ...
    def set_chunk_parallel_send_size(self, size: int) -> None: ...
```

这使得我们可以实现自定义的 `IChannel`，将 SPU 的通信委托给 MPLang 的现有通信层。

## Architecture

### Current Architecture (Dual-Channel)

```
┌─────────────┐     HTTP      ┌─────────────┐
│  Session/   │ ◄────────────► │  Session/   │
│  Simulator  │               │  Simulator  │
│             │               │             │
│  ┌────────┐ │               │  ┌────────┐ │
│  │MPLang  │ │               │  │MPLang  │ │
│  │Comm    │ │               │  │Comm    │ │
│  └────────┘ │               │  └────────┘ │
│             │               │             │
│  ┌────────┐ │     BRPC      │  ┌────────┐ │
│  │SPU Link│ │ ◄────────────► │  │SPU Link│ │
│  │(BRPC)  │ │  (new port)   │  │(BRPC)  │ │
│  └────────┘ │               │  └────────┘ │
└─────────────┘               └─────────────┘
   Node 0                         Node 1
   :8100                          :8101
   :8200 (SPU)                    :8201 (SPU)
```

### Proposed Architecture (Channel-Reuse)

```
┌─────────────┐     HTTP      ┌─────────────┐
│  Session/   │ ◄────────────► │  Session/   │
│  Simulator  │               │  Simulator  │
│             │               │             │
│  ┌────────┐ │               │  ┌────────┐ │
│  │MPLang  │ │               │  │MPLang  │ │
│  │Comm    │ │               │  │Comm    │ │
│  └───▲────┘ │               │  └───▲────┘ │
│      │      │               │      │      │
│  ┌───┴────┐ │   (reuse      │  ┌───┴────┐ │
│  │MPLang  │ │    HTTP)      │  │MPLang  │ │
│  │Channel │ │               │  │Channel │ │
│  │(IChannel)│               │  │(IChannel)│
│  └───▲────┘ │               │  └───▲────┘ │
│      │      │               │      │      │
│  ┌───┴────┐ │               │  ┌───┴────┐ │
│  │SPU Link│ │               │  │SPU Link│ │
│  │(Custom)│ │               │  │(Custom)│ │
│  └────────┘ │               │  └────────┘ │
└─────────────┘               └─────────────┘
   Node 0                         Node 1
   :8100 (only)                   :8101 (only)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `CommunicatorBase` | 底层通信抽象（send/recv with msgbox） |
| `ThreadCommunicator` | 进程内内存通信（模拟器） |
| `HttpCommunicator` | 跨进程 HTTP 通信（分布式） |
| `MPLangChannel` | **新增**：IChannel 适配器，桥接 Comm → SPU |
| `LinkCommunicator` | SPU link 上下文封装，支持 BRPC/Mem/Channels 三种模式 |

## API Surface

### 1. New Class: `MPLangChannel`

**Location**: `mplang/v1/runtime/spu_channel.py`

```python
from typing import TYPE_CHECKING
import spu.libspu as libspu
from mplang.v1.core.comm import CommunicatorBase

if TYPE_CHECKING:
    from mplang.v1.core.mask import Mask


class MPLangChannel(libspu.link.IChannel):
    """Bridge MPLang CommunicatorBase to SPU IChannel interface.
    
    This adapter allows SPU to use MPLang's existing communication layer
    (ThreadCommunicator or HttpCommunicator) instead of creating separate
    BRPC connections.
    
    Note: Each MPLangChannel represents a channel to ONE peer rank.
    SPU link.Context requires N-1 channels (one for each peer in the SPU mask).
    """
    
    def __init__(
        self, 
        comm: CommunicatorBase, 
        local_rank: int,
        peer_rank: int,
        tag_prefix: str = "spu",
    ):
        """Initialize channel to a specific peer.
        
        Args:
            comm: MPLang communicator instance
            local_rank: Global rank of this party
            peer_rank: Global rank of the peer party
            tag_prefix: Prefix for all tags (to avoid collision with non-SPU traffic)
        """
        super().__init__()
        self._comm = comm
        self._local_rank = local_rank
        self._peer_rank = peer_rank
        self._tag_prefix = tag_prefix
        
    def _make_key(self, tag: str) -> str:
        """Create unique key for MPLang comm (避免与其他流量冲突)"""
        return f"{self._tag_prefix}:{tag}"
        
    def send(self, tag: str, data: bytes) -> None:
        """Send bytes to peer (synchronous in SPU semantics)"""
        key = self._make_key(tag)
        # MPLang 的 send 需要 Value 封装，这里直接传 bytes
        # 需要 comm 支持 bytes 或包装为轻量 Value
        self._comm.send(self._peer_rank, key, data)
        
    def recv(self, tag: str) -> bytes:
        """Receive bytes from peer (blocking)"""
        key = self._make_key(tag)
        data = self._comm.recv(self._peer_rank, key)
        # 假设 comm 返回 bytes（或从 Value 中提取）
        if isinstance(data, bytes):
            return data
        # 如果是 Value，需要提取 bytes
        raise NotImplementedError("Value unwrapping not implemented")
        
    def send_async(self, tag: str, data: bytes) -> None:
        """Async send (MPLang's send is already async at network layer)"""
        self.send(tag, data)  # HttpCommunicator 的 httpx.put 已是异步
        
    def send_async_throttled(self, tag: str, data: bytes) -> None:
        """Throttled async send"""
        self.send_async(tag, data)
        
    def test_send(self, tag: str) -> bool:
        """Test if send buffer is available (always true for MPLang)"""
        return True
        
    def test_recv(self, tag: str) -> bool:
        """Test if data is available for recv (non-blocking)"""
        key = self._make_key(tag)
        # 需要 comm 支持非阻塞检测
        # 可以扩展 CommunicatorBase 添加 has_message(frm, key) 方法
        return False  # 占位实现
        
    def wait_link_task_finish(self) -> None:
        """Wait for all pending async tasks (no-op for MPLang)"""
        pass
        
    def abort(self) -> None:
        """Abort communication (清理资源)"""
        # 可以通知 comm 清理该 session 的消息
        pass
        
    def set_throttle_window_size(self, size: int) -> None:
        """Set throttle window (no-op)"""
        pass
        
    def set_chunk_parallel_send_size(self, size: int) -> None:
        """Set chunk size (no-op)"""
        pass
```

### 2. Modified: `LinkCommunicator`

**Location**: `mplang/v1/runtime/link_comm.py`

```python
class LinkCommunicator:
    """Minimal wrapper for libspu link context.
    
    Supports three modes:
    1. BRPC: Production mode with separate BRPC ports (legacy)
    2. Mem: In-memory links for testing (legacy)
    3. Channels: Reuse MPLang communicator via IChannel bridge (NEW)
    """
    
    def __init__(
        self,
        rank: int,
        addrs: list[str] | None = None,
        *,
        mem_link: bool = False,
        comm: CommunicatorBase | None = None,
        spu_mask: Mask | None = None,
    ):
        """Initialize link communicator for SPU.
        
        Args:
            rank: Global rank of this party
            addrs: List of addresses for all SPU parties (BRPC/Mem mode)
            mem_link: If True, use in-memory link (legacy)
            comm: MPLang communicator to reuse (Channels mode)
            spu_mask: SPU parties mask (required for Channels mode)
        
        Mode selection:
            - If comm is provided: Channels mode (NEW)
            - Elif mem_link is True: Mem mode
            - Else: BRPC mode
        """
        self._rank = rank
        
        # Mode 1: Channels (NEW) - Reuse MPLang communicator
        if comm is not None:
            if spu_mask is None:
                raise ValueError("spu_mask required when using comm")
            if rank not in spu_mask:
                raise ValueError(f"rank {rank} not in spu_mask {spu_mask}")
                
            # Create channels to all other SPU parties
            from mplang.v1.runtime.spu_channel import MPLangChannel
            
            channels = []
            for peer_rank in spu_mask:
                if peer_rank == rank:
                    continue  # Skip self
                channel = MPLangChannel(comm, rank, peer_rank)
                channels.append(channel)
            
            # Create link context with custom channels
            desc = libspu.link.Desc()  # type: ignore
            desc.recv_timeout_ms = 100 * 1000
            # Note: No need to add_party when using create_with_channels
            
            # Convert global rank to relative rank within SPU mask
            rel_rank = Mask(spu_mask).global_to_relative_rank(rank)
            self.lctx = libspu.link.create_with_channels(desc, rel_rank, channels)
            self._world_size = spu_mask.num_parties()
            
            logging.info(
                f"LinkCommunicator initialized with MPLangChannel: "
                f"rank={rank}, rel_rank={rel_rank}, spu_mask={spu_mask}"
            )
        
        # Mode 2 & 3: BRPC or Mem (legacy)
        else:
            if addrs is None:
                raise ValueError("addrs required for BRPC/Mem mode")
            self._world_size = len(addrs)
            
            desc = libspu.link.Desc()  # type: ignore
            desc.recv_timeout_ms = 100 * 1000
            desc.http_max_payload_size = 32 * 1024 * 1024
            for rank_idx, addr in enumerate(addrs):
                desc.add_party(f"P{rank_idx}", addr)
            
            if mem_link:
                self.lctx = libspu.link.create_mem(desc, self._rank)
                logging.info(
                    f"LinkCommunicator initialized with Mem: "
                    f"rank={rank}, addrs={addrs}"
                )
            else:
                self.lctx = libspu.link.create_brpc(desc, self._rank)
                logging.info(
                    f"LinkCommunicator initialized with BRPC: "
                    f"rank={rank}, addrs={addrs}"
                )
    
    # ... (rest of methods unchanged)
```

### 3. Modified: `Simulator.__init__`

**Location**: `mplang/v1/runtime/simulation.py`

```python
# Around line 130-140, replace SPU link creation:

# OLD:
# spu_addrs = [f"P{spu_rank}" for spu_rank in spu_mask]
# self._spu_link_ctxs: list[LinkCommunicator | None] = [None] * world_size
# link_ctx_list = [
#     LinkCommunicator(idx, spu_addrs, mem_link=True)
#     for idx in range(spu_mask.num_parties())
# ]
# for g_rank in range(world_size):
#     if g_rank in spu_mask:
#         rel = Mask(spu_mask).global_to_relative_rank(g_rank)
#         self._spu_link_ctxs[g_rank] = link_ctx_list[rel]

# NEW:
self._spu_link_ctxs: list[LinkCommunicator | None] = [None] * world_size
for g_rank in range(world_size):
    if g_rank in spu_mask:
        # Reuse ThreadCommunicator instead of creating separate mem_link
        link_ctx = LinkCommunicator(
            rank=g_rank,
            comm=self._comms[g_rank],  # Reuse!
            spu_mask=spu_mask,
        )
        self._spu_link_ctxs[g_rank] = link_ctx
```

### 4. Modified: `Session._seed_spu_env`

**Location**: `mplang/v1/runtime/session.py`

```python
# Around line 185-205, replace SPU link creation:

# OLD:
# link_ctx = None
# SPU_PORT_OFFSET = 100
# if self.is_spu_party:
#     spu_addrs: list[str] = []
#     for r, addr in enumerate(self.cluster_spec.endpoints):
#         if r in self.spu_mask:
#             parsed = urlparse(addr)
#             new_addr = f"{parsed.hostname}:{parsed.port + SPU_PORT_OFFSET}"
#             spu_addrs.append(new_addr)
#     rel_index = sum(1 for r in range(self.rank) if r in self.spu_mask)
#     link_ctx = g_link_factory.create_link(rel_index, spu_addrs)

# NEW:
link_ctx = None
if self.is_spu_party:
    # Reuse HttpCommunicator instead of creating separate BRPC connection
    # Assumption: Session has self.comm attribute
    link_ctx = LinkCommunicator(
        rank=self.rank,
        comm=self.comm,  # Reuse existing HTTP communicator!
        spu_mask=self.spu_mask,
    )
```

### 5. Optional Enhancement: `CommunicatorBase`

**Location**: `mplang/v1/core/comm.py`

Add support for non-blocking check and direct bytes handling:

```python
class CommunicatorBase(ICommunicator):
    # ... existing code ...
    
    def has_message(self, frm: int, key: str) -> bool:
        """Non-blocking check if message is available.
        
        For IChannel.test_recv() support.
        """
        with self._cond:
            mkey = (frm, key)
            return mkey in self._msgboxes
    
    def peek_message(self, frm: int, key: str) -> Any | None:
        """Non-blocking peek at message without consuming.
        
        Returns None if message not available.
        """
        with self._cond:
            mkey = (frm, key)
            return self._msgboxes.get(mkey)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

- [ ] Create `mplang/v1/runtime/spu_channel.py` with `MPLangChannel` class
- [ ] Add Channels mode to `LinkCommunicator.__init__`
- [ ] Add unit tests for `MPLangChannel` (mock SPU send/recv)

### Phase 2: Simulator Integration (Week 1)

- [ ] Modify `Simulator.__init__` to use Channels mode
- [ ] Run existing SPU tests (`tests/v1/kernels/test_spu.py`)
- [ ] Verify no BRPC ports created in simulation

### Phase 3: Session/Driver Integration (Week 2)

- [ ] Ensure `Session` has accessible `comm` attribute
- [ ] Modify `Session._seed_spu_env` to use Channels mode
- [ ] Test with distributed setup (3-party HTTP cluster)

### Phase 4: Optional Enhancements (Week 2)

- [ ] Add `has_message()` to `CommunicatorBase` for `test_recv()`
- [ ] Performance benchmarking (BRPC vs HTTP for SPU traffic)
- [ ] Add configuration flag to control mode selection

### Phase 5: Migration & Deprecation (Week 3)

- [ ] Make Channels mode the default (if performance acceptable)
- [ ] Add deprecation warning for BRPC mode
- [ ] Update documentation and examples
- [ ] Remove `SPU_PORT_OFFSET` logic from session.py

## Testing Strategy

### Unit Tests

```python
# tests/v1/runtime/test_spu_channel.py

def test_mplang_channel_send_recv():
    """Test basic send/recv through MPLangChannel"""
    world_size = 2
    comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
    for c in comms: c.set_peers(comms)
    
    # Create channels: rank0 -> rank1
    ch0 = MPLangChannel(comms[0], local_rank=0, peer_rank=1)
    ch1 = MPLangChannel(comms[1], local_rank=1, peer_rank=0)
    
    # Test send/recv
    data = b"hello spu"
    ch0.send("tag1", data)
    received = ch1.recv("tag1")
    assert received == data

def test_link_communicator_channels_mode():
    """Test LinkCommunicator with Channels mode"""
    world_size = 3
    spu_mask = Mask.from_ranks([0, 1, 2])
    comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
    for c in comms: c.set_peers(comms)
    
    links = [
        LinkCommunicator(rank=i, comm=comms[i], spu_mask=spu_mask)
        for i in range(world_size)
    ]
    
    assert all(link.world_size == 3 for link in links)
    assert all(link.rank == i for i, link in enumerate(links))
```

### Integration Tests

- Run all existing SPU kernel tests with Channels mode enabled
- Compare outputs with BRPC mode (should be identical)
- Test error handling (abort, timeout, etc.)

### Performance Tests

```python
# Benchmark: BRPC vs HTTP for typical SPU workload
# Metrics: latency, throughput, CPU usage, memory
```

## Migration & Compatibility

### Backward Compatibility

- **Keep BRPC mode available** as fallback (controlled by config)
- Use feature detection: `hasattr(libspu.link, 'create_with_channels')`
- Graceful degradation if old libspu version detected

### Configuration

Add to cluster config YAML:

```yaml
devices:
  SP0:
    kind: "SPU"
    config:
      protocol: "ABY3"
      field: "FM64"
      # New option (default: true)
      use_mplang_channel: true
```

Or environment variable:

```bash
export MPLANG_SPU_USE_CHANNELS=1  # Enable (default)
export MPLANG_SPU_USE_CHANNELS=0  # Disable (use BRPC)
```

### Code Example

```python
# Auto-detection and fallback
if use_mplang_channel and hasattr(libspu.link, 'create_with_channels'):
    link_ctx = LinkCommunicator(rank=..., comm=..., spu_mask=...)
else:
    # Fallback to BRPC
    link_ctx = LinkCommunicator(rank=..., addrs=..., mem_link=False)
```

## Benefits

### Deployment Simplification

**Before**:
```yaml
nodes:
  - endpoint: "http://127.0.0.1:8100"
    # SPU implicitly uses :8200 (requires firewall rule)
  - endpoint: "http://127.0.0.1:8101"
    # SPU implicitly uses :8201
```

**After**:
```yaml
nodes:
  - endpoint: "http://127.0.0.1:8100"  # That's it!
  - endpoint: "http://127.0.0.1:8101"
```

### Resource Savings

- **50% fewer ports** (1 per node instead of 2)
- **~30% less memory** (single HTTP pool vs HTTP + BRPC)
- **Simpler firewall rules**

### Developer Experience

- Unified logging: all communication via same channel
- Easier debugging: single protocol stack
- Consistent error handling

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| HTTP slower than BRPC | Medium | Benchmark first; keep BRPC as option |
| libspu version incompatibility | High | Feature detection + graceful fallback |
| Serialization overhead | Low | Use zero-copy where possible |
| Debugging complexity | Low | Add detailed logging in MPLangChannel |
| Message collision | Medium | Use unique tag prefixes (`spu:*`) |

## Alternatives Considered

### 1. Keep Dual-Channel (No Change)

**Pros**: No implementation cost, proven stable  
**Cons**: Continues deployment complexity

### 2. Implement Custom BRPC Server in Python

**Pros**: Same performance as C++ BRPC  
**Cons**: Massive implementation effort, duplicates work

### 3. Use gRPC Instead of HTTP

**Pros**: Better performance than HTTP  
**Cons**: Still requires separate port, doesn't reuse existing infra

## Open Questions

1. **Performance**: Is HTTP latency acceptable for SPU workloads?
   - Action: Run benchmarks with realistic workloads

2. **Value Serialization**: Should we create `RawBytesValue` or use existing types?
   - Action: Test with TensorValue wrapping bytes

3. **Non-blocking recv**: How to efficiently implement `test_recv()`?
   - Action: Add `has_message()` to CommunicatorBase

4. **Error Recovery**: How to handle partial failures in channels?
   - Action: Define error handling contract in MPLangChannel

## References

- libspu source: `yacl/link/context.h`, `yacl/link/channel.h`
- MPLang v1 comm: `mplang/v1/core/comm.py`
- Session SPU setup: `mplang/v1/runtime/session.py:_seed_spu_env`
- Simulator SPU setup: `mplang/v1/runtime/simulation.py:__init__`

## Changelog

- 2025-12-29: Initial design draft (zhsu)
