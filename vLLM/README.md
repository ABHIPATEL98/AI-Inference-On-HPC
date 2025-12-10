# vLLM Deployment Guide: Single-Node & Multi-Node Inference on HPC

This documentation outlines the procedures for installing, configuring, and deploying the vLLM (Versatile Large Language Model) inference engine on High-Performance Computing (HPC) clusters. It covers three specific architectural configurations:

1.  **Single Node / Single GPU**
2.  **Single Node / Multi-GPU (Dense)**
3.  **Multi-Node / Multi-GPU (Distributed via Ray)**

-----

## 1\. Environmental Setup & Installation

Before requesting compute resources, you must establish the software environment.

### **Step 1: Create Conda Environment**

Execute the following commands to create a clean Python 3.10 environment.

```bash
conda create --name vllm python=3.10 --yes
conda activate vllm
```

### **Step 2: Install vLLM**

Install the specific version of vLLM required for this deployment.

```bash
pip install vllm==0.11.0
```

> **Note:** Ensure your CUDA drivers are compatible with the PyTorch version installed by vLLM. You may need to load CUDA modules provided by your HPC administrator (e.g., `module load cuda/12.1`).

-----

## 2\. Resource Allocation & Networking

### **Step 1: Request Compute Nodes**

Acquire GPU resources using the Slurm workload manager. You may run interactively or submit a batch job.

  * **Interactive Mode:** `srun --pty --gres=gpu:1 -t 01:00:00 /bin/bash`
  * **Batch Mode:** Submit a `.slurm` script using `sbatch`.

### **Step 2: Access & Network Configuration**

Once the job starts, SSH into the allocated node (e.g., `hopper01`) and identify the **Infiniband (IB)** IP address. This is critical for low-latency communication in multi-node setups.

```bash
ssh hopper01
ip a
```

> **Task:** Look for the high-speed network interface (often labeled `ib0`, `ib1`, or similar). Note down the IP address (e.g., `172.17.1.1`).

-----

## 3\. Inference Configurations

### Configuration A: Single Node / Single GPU

**Use Case:** Debugging, development, or serving smaller models (e.g., Llama-3.2-3B).

Run the following command inside the GPU node:

```bash
vllm serve /home/abhishekp/llm_model/Llama-3.2-3B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name Llama-3.2-3B-Instruct \
  --dtype float16
```

#### **Parameter Explanation:**

  * `--host 0.0.0.0`: Allows the server to accept connections from outside the localhost (required for accessing the API from a login node or web client).
  * `--served-model-name`: Sets the identifier used in the API call `model` field.
  * `--dtype float16`: Forces the model weights to load in half-precision (FP16) to reduce memory usage.

-----

### Configuration B: Single Node / Multi-GPU (Dense)

**Use Case:** Serving larger models that fit on one server but require multiple GPUs for memory or faster throughput.

**Requirements:** Adjust the `--tensor-parallel-size` argument to match the number of GPUs you wish to utilize (e.g., 4 GPUs).

```bash
vllm serve /home/abhishekp/llm_model/Llama-3.2-3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --served-model-name Llama-3.2-3B-Instruct \
  --dtype float16
```

#### **Parameter Explanation:**

  * `--tensor-parallel-size 4`: Splits the model layers across 4 GPUs within the same node. Each GPU handles a portion of the calculation for every token generated.

-----

### Configuration C: Multi-Node / Multi-GPU (Distributed)

**Use Case:** Serving massive models (e.g., Llama-3-70B, Grok-1) that cannot fit into the VRAM of a single node. This configuration utilizes **Ray** for cluster orchestration.

**Prerequisites:** Request **minimum 2 GPU nodes** via Slurm.

#### **Phase 1: Ray Cluster Initialization**

**1. On the Leader Node (e.g., hopper01 - 172.17.1.1)**
Export the Infiniband IP of the current node and start the Ray leader process.

```bash
export VLLM_HOST_IP=172.17.1.1

./multi-node-serving.sh leader \
  --ray_port=6379 \
  --ray_cluster_size=2 \
  --node-ip-address=172.17.1.1 \
  --num-gpus=4 \
  --temp-dir=/tmp/ray \
  --include-dashboard=False
```

**Leader Parameter Definitions:**

  * `leader`: Instructs the script to initialize a Ray **Head** node.
  * `--ray_port=6379`: The port used for Redis communication between nodes.
  * `--ray_cluster_size=2`: The total number of nodes expected in this Ray cluster.
  * `--node-ip-address`: The Infiniband IP of the Leader node.
  * `--num-gpus=4`: Number of GPUs to manage on *this specific node*.
  * `--include-dashboard=False`: Disables the Ray web UI (saves resources/security).

<br>

**2. On the Worker Node (e.g., hopper02 - 172.17.1.3)**
Export the Infiniband IP of the *worker* node, then connect it to the leader.

```bash
export VLLM_HOST_IP=172.17.1.3

./multi-node-serving.sh worker \
  --ray_address=172.17.1.1 \
  --ray_port=6379 \
  --node-ip-address=172.17.1.3 \
  --num-gpus=4 \
  --temp-dir=/tmp/ray
```

**Worker Parameter Definitions:**

  * `worker`: Instructs the script to start a Ray **Worker** node.
  * `--ray_address=172.17.1.1`: The IP address of the **Leader** node (must match the leader's IB IP).
  * `--node-ip-address`: The Infiniband IP of *this* Worker node.

<br>

**3. Verification**
On either node, check the status of the cluster:

```bash
ray status
```

> **Success Criteria:** You should see a summary listing the total GPUs (e.g., 8 GPUs if using 2 nodes with 4 GPUs each), total CPU cores, and connected Node IDs.

-----

#### **Phase 2: Launching Distributed Inference**

Once the Ray cluster is active, run the vLLM serving command **on the Leader Node**.

```bash
VLLM_USE_SYMMETRIC_MEMORY=0 vllm serve /home/abhishekp/llm_model/Meta-Llama-3.1-70B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --served-model-name meta-llama-3.1-70b \
  --api-key token-abc123 \
  --dtype auto \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 2048 \
  --gpu-memory-utilization 0.95
```

#### **Inference Parameter Definitions:**

  * **`VLLM_USE_SYMMETRIC_MEMORY=0`**: Environment variable to disable symmetric memory mapping. This is often required in multi-node setups to prevent NCCL failures if memory addresses differ across nodes.
  * **`--tensor-parallel-size 4`**: The number of GPUs used for Tensor Parallelism (TP). This usually matches the number of GPUs **per node**.
  * **`--pipeline-parallel-size 2`**: The number of Pipeline Parallel (PP) stages. This essentially splits the model layers across nodes.
      * *Calculation:* Total GPUs used = $TP \times PP = 4 \times 2 = 8$ GPUs.
  * **`--max-num-batched-tokens 32768`**: The maximum number of tokens vLLM will process in a single iteration (batch). Higher values increase throughput but use more KV-Cache memory.
  * **`--max-num-seqs 2048`**: Maximum number of concurrent sequences (requests) the engine will handle.
  * **`--gpu-memory-utilization 0.95`**: vLLM will reserve 95% of available GPU VRAM. The remaining 5% is left for activation overhead and PyTorch context.
  * **`--api-key`**: Sets a security token required to query the API.

-----

## 4\. API Usage & Testing

Wait until the application logs indicate **"Application Startup Complete"**. You can then perform inference using standard OpenAI-compatible HTTP requests.

### **Sample CURL Request**

```bash
curl http://172.17.1.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "meta-llama-3.1-70b",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert HPC assistant helping with technical documentation."
      },
      {
        "role": "user",
        "content": "How do I submit a SLURM job?"
      }
    ],
    "temperature": 0.0,
    "max_tokens": 4096
  }'
```

### **Troubleshooting Tips**
**1. Advanced Network/NCCL Debugging:**
If you encounter NCCL timeouts, initialization failures, or communication errors between nodes, you may need to explicitly define the network interface and debug parameters in your `~/.bashrc` or export them before running the script.

Add the following environment variables:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib1
export NCCL_IB_GID_INDEX=3
````

  * `NCCL_DEBUG=INFO`: Provides verbose logs to diagnose communication hang-ups.
  * `NCCL_IB_HCA=mlx5`: Forces usage of Mellanox ConnectX adapters (verify your HCA name with `ibdev2netdev`).
  * `NCCL_SOCKET_IFNAME`: Specifies the explicit interface for socket communication (ensure this matches your `ip a` output).


2.  **OOM (Out Of Memory):** Reduce `--gpu-memory-utilization` to `0.90` or reduce `--max-num-batched-tokens`.
3.  **Connection Refused:** Ensure you are using the correct IP address and that the `vllm serve` command includes `--host 0.0.0.0`.

-----

*Document created for High-Performance Computing Cluster Usage.*
