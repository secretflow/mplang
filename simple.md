# Millionaire Device Analysis


## Cluster Specification


```json

{
  "nodes": [
    {
      "name": "node_0",
      "rank": 0,
      "endpoint": "127.0.0.1:61930"
    },
    {
      "name": "node_1",
      "rank": 1,
      "endpoint": "127.0.0.1:61931"
    },
    {
      "name": "node_2",
      "rank": 2,
      "endpoint": "127.0.0.1:61932"
    }
  ],
  "devices": {
    "P0": {
      "kind": "PPU",
      "members": [
        "node_0"
      ]
    },
    "P1": {
      "kind": "PPU",
      "members": [
        "node_1"
      ]
    },
    "SP0": {
      "kind": "SPU",
      "members": [
        "node_0",
        "node_1",
        "node_2"
      ]
    },
    "TEE0": {
      "kind": "TEE",
      "members": [
        "node_2"
      ]
    }
  }
}

```

## Compiler IR (text)


```

() {
  %0 = peval() {fn_type=mlir.stablehlo, fn_name=randint, rmask=0x1} : i64<1>{_devid_="P0"}
  %1 = peval() {fn_type=mlir.stablehlo, fn_name=randint, rmask=0x2} : i64<2>{_devid_="P1"}
  %2 = peval(%0) {fn_type=crypto.pack, rmask=0x1} : u8[8]<1>
  %3 = peval() {fn_type=crypto.kem_keygen, rmask=0x1} : (u8[32]<1>, u8[32]<1>)
  %4 = peval() {fn_type=crypto.kem_keygen, rmask=0x4} : (u8[32]<4>, u8[32]<4>)
  %5 = peval(%4:1) {fn_type=tee.quote, rmask=0x4} : u8[33]<4>
  %6 = pshfl_s(%5) {pmask=1, src_ranks=[2]} : u8[33]<1>
  %7 = peval(%6) {fn_type=tee.attest, rmask=0x1} : u8[32]<1>
  %8 = peval(%3:0, %7) {fn_type=crypto.kem_derive, rmask=0x1} : u8[32]<1>
  %9 = peval(%8) {fn_type=crypto.hkdf, rmask=0x1} : u8[32]<1>
  %10 = peval(%2, %9) {fn_type=crypto.enc, rmask=0x1} : u8[20]<1>
  %11 = pshfl_s(%10) {pmask=4, src_ranks=[0]} : u8[20]<4>
  %12 = pshfl_s(%3:1) {pmask=4, src_ranks=[0]} : u8[32]<4>
  %13 = peval(%4:0, %12) {fn_type=crypto.kem_derive, rmask=0x4} : u8[32]<4>
  %14 = peval(%13) {fn_type=crypto.hkdf, rmask=0x4} : u8[32]<4>
  %15 = peval(%11, %14) {fn_type=crypto.dec, rmask=0x4} : u8[8]<4>
  %16 = peval(%15) {fn_type=crypto.unpack, rmask=0x4} : i64<4>{_devid_="TEE0"}
  %17 = peval(%1) {fn_type=crypto.pack, rmask=0x2} : u8[8]<2>
  %18 = peval() {fn_type=crypto.kem_keygen, rmask=0x2} : (u8[32]<2>, u8[32]<2>)
  %19 = peval() {fn_type=crypto.kem_keygen, rmask=0x4} : (u8[32]<4>, u8[32]<4>)
  %20 = peval(%19:1) {fn_type=tee.quote, rmask=0x4} : u8[33]<4>
  %21 = pshfl_s(%20) {pmask=2, src_ranks=[2]} : u8[33]<2>
  %22 = peval(%21) {fn_type=tee.attest, rmask=0x2} : u8[32]<2>
  %23 = peval(%18:0, %22) {fn_type=crypto.kem_derive, rmask=0x2} : u8[32]<2>
  %24 = peval(%23) {fn_type=crypto.hkdf, rmask=0x2} : u8[32]<2>
  %25 = peval(%17, %24) {fn_type=crypto.enc, rmask=0x2} : u8[20]<2>
  %26 = pshfl_s(%25) {pmask=4, src_ranks=[1]} : u8[20]<4>
  %27 = pshfl_s(%18:1) {pmask=4, src_ranks=[1]} : u8[32]<4>
  %28 = peval(%19:0, %27) {fn_type=crypto.kem_derive, rmask=0x4} : u8[32]<4>
  %29 = peval(%28) {fn_type=crypto.hkdf, rmask=0x4} : u8[32]<4>
  %30 = peval(%26, %29) {fn_type=crypto.dec, rmask=0x4} : u8[8]<4>
  %31 = peval(%30) {fn_type=crypto.unpack, rmask=0x4} : i64<4>{_devid_="TEE0"}
  %32 = peval(%16, %31) {fn_type=mlir.stablehlo, fn_name=<lambda>, rmask=0x4} : bool<4>{_devid_="TEE0"}
  %33 = peval(%32) {fn_type=crypto.pack, rmask=0x4} : u8[1]<4>
  %34 = peval(%33, %14) {fn_type=crypto.enc, rmask=0x4} : u8[13]<4>
  %35 = pshfl_s(%34) {pmask=1, src_ranks=[2]} : u8[13]<1>
  %36 = peval(%35, %9) {fn_type=crypto.dec, rmask=0x1} : u8[1]<1>
  %37 = peval(%36) {fn_type=crypto.unpack, rmask=0x1} : bool<1>{_devid_="P0"}
  %38 = tuple(%0, %1, %32, %37) : (i64<1>{_devid_="P0"}, i64<2>{_devid_="P1"}, bool<4>{_devid_="TEE0"}, bool<1>{_devid_="P0"})
  return %38
}

```

## Graph Structure Analysis


```

GraphProto structure analysis:
- Version: 1.0.0
- Number of nodes: 75
- Number of outputs: 4
- Graph attributes: 1

Node breakdown by operation type:
- access: 35 nodes
- eval: 31 nodes
- func_def: 1 nodes
- shfl_s: 7 nodes
- tuple: 1 nodes

Output variables:
- Output 0: %74:0
- Output 1: %74:1
- Output 2: %74:2
- Output 3: %74:3

```

## Mermaid Sequence Diagram

```mermaid

sequenceDiagram
participant P0
participant P1
participant P2
P0-->>P0: randint %0
P1-->>P1: randint %2
P0-->>P0: crypto.pack %4
P0-->>P0: crypto.kem_keygen %6 -> 2 outs
P2-->>P2: crypto.kem_keygen %8 -> 2 outs
P2-->>P2: tee.quote %10
P2->>P0: %12
P0-->>P0: tee.attest %13
P0-->>P0: crypto.kem_derive %15
P0-->>P0: crypto.hkdf %17
P0-->>P0: crypto.enc %19
P0->>P2: %21
P0->>P2: %24
P2-->>P2: crypto.kem_derive %25
P2-->>P2: crypto.hkdf %27
P2-->>P2: crypto.dec %29
P2-->>P2: crypto.unpack %31
P1-->>P1: crypto.pack %33
P1-->>P1: crypto.kem_keygen %35 -> 2 outs
P2-->>P2: crypto.kem_keygen %37 -> 2 outs
P2-->>P2: tee.quote %39
P2->>P1: %41
P1-->>P1: tee.attest %42
P1-->>P1: crypto.kem_derive %44
P1-->>P1: crypto.hkdf %46
P1-->>P1: crypto.enc %48
P1->>P2: %50
P1->>P2: %53
P2-->>P2: crypto.kem_derive %54
P2-->>P2: crypto.hkdf %56
P2-->>P2: crypto.dec %58
P2-->>P2: crypto.unpack %60
P2-->>P2: <lambda> %62
P2-->>P2: crypto.pack %64
P2-->>P2: crypto.enc %66
P2->>P0: %68
P0-->>P0: crypto.dec %69
P0-->>P0: crypto.unpack %71

```

## Mermaid Flowchart (DAG)

```mermaid

graph TB;

    subgraph P0
        n0["randint @P0"]
        n4["crypto.pack @P0"]
        n6["crypto.kem_keygen/2 @P0"]
        n12["shfl_s @P0"]
        n13["tee.attest @P0"]
        n15["crypto.kem_derive @P0"]
        n17["crypto.hkdf @P0"]
        n19["crypto.enc @P0"]
        n68["shfl_s @P0"]
        n69["crypto.dec @P0"]
        n71["crypto.unpack @P0"]
    end

    subgraph P1
        n2["randint @P1"]
        n33["crypto.pack @P1"]
        n35["crypto.kem_keygen/2 @P1"]
        n41["shfl_s @P1"]
        n42["tee.attest @P1"]
        n44["crypto.kem_derive @P1"]
        n46["crypto.hkdf @P1"]
        n48["crypto.enc @P1"]
    end

    subgraph P2
        n8["crypto.kem_keygen/2 @P2"]
        n10["tee.quote @P2"]
        n21["shfl_s @P2"]
        n24["shfl_s @P2"]
        n25["crypto.kem_derive @P2"]
        n27["crypto.hkdf @P2"]
        n29["crypto.dec @P2"]
        n31["crypto.unpack @P2"]
        n37["crypto.kem_keygen/2 @P2"]
        n39["tee.quote @P2"]
        n50["shfl_s @P2"]
        n53["shfl_s @P2"]
        n54["crypto.kem_derive @P2"]
        n56["crypto.hkdf @P2"]
        n58["crypto.dec @P2"]
        n60["crypto.unpack @P2"]
        n62["<lambda> @P2"]
        n64["crypto.pack @P2"]
        n66["crypto.enc @P2"]
    end

    n0 --> n4
    n10 --> n12
    n12 --> n13
    n13 --> n15
    n15 --> n17
    n17 --> n19
    n17 --> n69
    n19 --> n21
    n2 --> n33
    n21 --> n29
    n24 --> n25
    n25 --> n27
    n27 --> n29
    n27 --> n66
    n29 --> n31
    n31 --> n62
    n33 --> n48
    n35 --> n44
    n35 --> n53
    n37 --> n39
    n37 --> n54
    n39 --> n41
    n4 --> n19
    n41 --> n42
    n42 --> n44
    n44 --> n46
    n46 --> n48
    n48 --> n50
    n50 --> n58
    n53 --> n54
    n54 --> n56
    n56 --> n58
    n58 --> n60
    n6 --> n15
    n6 --> n24
    n60 --> n62
    n62 --> n64
    n64 --> n66
    n66 --> n68
    n68 --> n69
    n69 --> n71
    n8 --> n10
    n8 --> n25
    linkStyle 1 stroke:#ff6a00,stroke-width:2px;
    linkStyle 7 stroke:#ff6a00,stroke-width:2px;
    linkStyle 18 stroke:#ff6a00,stroke-width:2px;
    linkStyle 21 stroke:#ff6a00,stroke-width:2px;
    linkStyle 27 stroke:#ff6a00,stroke-width:2px;
    linkStyle 34 stroke:#ff6a00,stroke-width:2px;
    linkStyle 38 stroke:#ff6a00,stroke-width:2px;

```
