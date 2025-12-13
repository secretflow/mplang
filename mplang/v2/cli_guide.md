# MPLang v2 CLI Workflow Guide

This document describes how to use the MPLang v2 Command Line Interface (CLI) to manage local clusters, submit jobs, and inspect runtime state. This workflow is designed for development and debugging.

## Prerequisites

Ensure you have `mplang` installed in your environment.

```bash
source .venv/bin/activate
# or
uv sync
```

## Workflow Overview

The recommended development workflow involves using two terminal windows:
1.  **Terminal 1**: Runs the cluster (servers).
2.  **Terminal 2**: Submits jobs and queries cluster status.

## Step-by-Step Guide

### 1. Generate Cluster Configuration

First, generate a `cluster.yaml` file. This defines the topology of your MPLang cluster.

```bash
# Generate a config for 2 workers starting at port 8100
python -m mplang.v2.cli config gen -w 2 -p 8100 -o cluster.yaml
```

### 2. Start the Cluster (Terminal 1)

In your **first terminal**, start the cluster using the `up` command. This will start the worker processes defined in your config.

```bash
# Terminal 1
python -m mplang.v2.cli up -c cluster.yaml
```

You should see logs indicating that workers have started (e.g., `[Worker 0] INFO: Started server process...`). Keep this terminal open.

### 3. Prepare a Job Script

Create a Python script (e.g., `my_job.py`) that defines the computation you want to run.

```python
# my_job.py
from mplang.v2.dialects import simp
import numpy as np

def main():
    print("Starting job...")

    # Create data on Party 0
    x = simp.constant((0,), np.array([1, 2, 3]))

    # Shuffle data to Party 1
    y = simp.shuffle_static(x, {1: 0})

    print("Computation graph constructed.")
    return y

if __name__ == "__main__":
    main()
```

### 4. Submit the Job (Terminal 2)

In your **second terminal**, use the `run` command to submit the script to the running cluster.

```bash
# Terminal 2
python -m mplang.v2.cli run -c cluster.yaml -f my_job.py
```

The CLI will connect to the driver, which orchestrates the execution across the workers.

### 5. Check Cluster Status

You can check the health and latency of your workers at any time.

```bash
# Terminal 2
python -m mplang.v2.cli status -c cluster.yaml
```

**Output Example:**
```text
Rank   | Endpoint                  | Status | Response Time
--------------------------------------------------------------------------------
0      | http://127.0.0.1:8100     | OK     | 0.002s
1      | http://127.0.0.1:8101     | OK     | 0.003s
```

### 6. Inspect Runtime Objects (Symbols)

To debug or verify intermediate results, you can list the objects currently stored in the memory of each worker.

```bash
# Terminal 2
python -m mplang.v2.cli objects -c cluster.yaml
```

**Output Example:**
```text
Rank   | Endpoint                  | Count  | Objects
--------------------------------------------------------------------------------
0      | http://127.0.0.1:8100     | 1      | 7f8b... (x)
1      | http://127.0.0.1:8101     | 1      | 3a2c... (y)
```

## Command Reference

| Command | Description | Usage |
| :--- | :--- | :--- |
| `config gen` | Generate cluster config file | `python -m mplang.v2.cli config gen -w <workers> -o <file>` |
| `up` | Start all workers locally | `python -m mplang.v2.cli up -c <config>` |
| `run` | Submit a job script | `python -m mplang.v2.cli run -c <config> -f <script>` |
| `status` | Check worker health | `python -m mplang.v2.cli status -c <config>` |
| `objects` | List objects on workers | `python -m mplang.v2.cli objects -c <config>` |
| `worker` | Start a single worker (prod) | `python -m mplang.v2.cli worker --rank <id> -c <config>` |
