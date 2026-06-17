# ==============================================================
# 基于人工智能的智能合约项目自动化恶意检测 完整代码（单文件）
# 对应论文：基于人工智能的智能合约项目自动化恶意检测技术
# 环境：Python3.9 + PyTorch2.1 + PyG2.5 + Transformers4.38 + PEFT0.10
# 功能：1. 图神经网络(GNN) 合约业务相似度识别  2. Llama2+LoRA 恶意交易检测
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

import pandas as pd
import numpy as np
import networkx as nx
import json
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------- 全局配置 & 随机种子 --------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[系统] 运行设备: {device}")

# 合约地址、操作码词表（论文操作码序列）
contract_addrs = [f"addr_{i}" for i in range(20)]
opcode_vocab = ["PUSH1", "SLOAD", "SWAP1", "POP", "JUMP1", "MSTORE", "CALL", "RETURN"]
opcode2idx = {op: idx for idx, op in enumerate(opcode_vocab)}
vocab_size = len(opcode_vocab)

# ========================== 模块一：图神经网络 业务相似度识别 ==========================
# -------------------------- 1.1 合约调用图数据构建 --------------------------
def generate_call_graph():
    """生成以太坊智能合约调用拓扑图（业务主干图）"""
    G = nx.DiGraph()
    node_num = random.randint(8, 18)
    nodes = random.sample(contract_addrs, node_num)
    G.add_nodes_from(nodes)
    # 生成合约调用边
    for u in nodes:
        for v in nodes:
            if u != v and random.random() < 0.3:
                G.add_edge(u, v)
    return G

def graph_to_pyg_data(nx_graph):
    """NetworkX图 转为 PyTorch Geometric 标准图数据"""
    nodes = list(nx_graph.nodes)
    node2idx = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # 节点特征：操作码随机嵌入
    x = torch.randn(num_nodes, vocab_size)
    # 边索引
    edge_list = []
    for u, v in nx_graph.edges:
        edge_list.append([node2idx[u], node2idx[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def build_graph_dataset(sample_num=200):
    """构建图数据集：标签0=DEX项目，1=借贷项目（业务分类）"""
    dataset = []
    for _ in range(sample_num):
        g = generate_call_graph()
        data = graph_to_pyg_data(g)
        label = torch.tensor([random.randint(0, 1)], dtype=torch.long)
        data.y = label
        dataset.append(data)
    return dataset

# 构建训练/测试集
graph_dataset = build_graph_dataset(200)
train_graph, test_graph = train_test_split(graph_dataset, test_size=0.2, random_state=seed)
train_loader = DataLoader(train_graph, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graph, batch_size=16, shuffle=False)
print(f"[图数据] 训练集: {len(train_graph)} 条 | 测试集: {len(test_graph)} 条")

# -------------------------- 1.2 残差GCN模型（论文核心网络） --------------------------
class ResGCNBlock(nn.Module):
    """残差GCN块，实现论文残差结构 y = F(x) + x"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.downsample(x)
        out = self.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        out += residual
        out = self.relu(out)
        return out

class BusinessGNN(nn.Module):
    """业务相似度识别主模型：三层残差GCN + TopK池化 + 全局读出层"""
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        # 三层残差GCN + TopK池化
        self.block1 = ResGCNBlock(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.7)

        self.block2 = ResGCNBlock(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.7)

        self.block3 = ResGCNBlock(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.7)

        # 分类头
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 第一层特征提取与池化
        x = self.block1(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        pool1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        # 第二层特征提取与池化
        x = self.block2(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        pool2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        # 第三层特征提取与池化
        x = self.block3(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)
        pool3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        # 特征聚合 & 分类
        graph_feat = pool1 + pool2 + pool3
        out = self.relu(self.lin1(graph_feat))
        out = self.lin2(out)
        return out

# 初始化GNN模型
gnn_model = BusinessGNN(in_channels=vocab_size, hidden_channels=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
print("[GNN模型] 业务相似度识别模型初始化完成")

# -------------------------- 1.3 GNN 训练与评估 --------------------------
def train_gnn():
    gnn_model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gnn_model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_gnn():
    gnn_model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = gnn_model(batch)
            pred = out.argmax(dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return acc, f1

# 执行GNN训练
print("\n===== 开始训练【业务相似度识别 GNN 模型】=====")
for epoch in range(30):
    loss = train_gnn()
    acc, f1 = test_gnn()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | 损失: {loss:.4f} | 准确率: {acc:.4f} | F1值: {f1:.4f}")
print("===== GNN 模型训练完成 =====")

# ========================== 模块二：Llama2 + LoRA 恶意交易检测 ==========================
# -------------------------- 2.1 操作码转自然语言 & 数据集构建 --------------------------
def opcode_to_text(opcode_seq):
    """操作码序列转换为自然语言描述（论文自然语段生成）"""
    text = "以太坊交易执行序列："
    text += " -> ".join(opcode_seq)
    text += "，该交易调用智能合约函数，执行存储、跳转、外部调用等操作。"
    return text

def generate_llm_dataset(sample_num=300):
    """生成大模型数据集：0=正常交易  1=恶意交易"""
    llm_data = []
    for _ in range(sample_num):
        seq_len = random.randint(6, 15)
        op_seq = random.choices(opcode_vocab, k=seq_len)
        text = opcode_to_text(op_seq)
        label = 1 if random.random() < 0.4 else 0
        # 构造训练Prompt模板
        prompt = f"""任务：判断以太坊交易是否为恶意交易。
交易描述：{text}
请输出结果：0(正常) 或 1(恶意)
答案：{label}"""
        llm_data.append({"text": prompt, "label": label})
    return pd.DataFrame(llm_data)

# 构建大模型数据集
llm_df = generate_llm_dataset(300)
train_llm, test_llm = train_test_split(llm_df, test_size=0.2, random_state=seed)
print(f"\n[LLM数据] 训练集: {len(train_llm)} 条 | 测试集: {len(test_llm)} 条")

# -------------------------- 2.2 LoRA 配置 & Llama2 模型加载 --------------------------
# 模型路径：替换为本地模型路径 或 HuggingFace 模型ID
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

# LoRA 低秩适配配置（论文LoRA微调）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 加载基础模型（8bit量化适配P100 16G显存）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)

# 注入LoRA适配器
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()
print("[LLM模型] Llama2 + LoRA 恶意检测模型初始化完成")

# -------------------------- 2.3 数据集格式化 & LoRA 训练 --------------------------
def tokenize_func(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 转换为HuggingFace Dataset格式
train_ds = Dataset.from_pandas(train_llm)
test_ds = Dataset.from_pandas(test_llm)
train_token = train_ds.map(tokenize_func, batched=True)
test_token = test_ds.map(tokenize_func, batched=True)

# 训练参数（适配论文硬件：Tesla P100）
training_args = TrainingArguments(
    output_dir="./llama2_lora_smartcontract",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

# 训练器
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=test_token
)

# 启动LoRA微调
print("\n===== 开始【Llama2 + LoRA】恶意检测模型微调 =====")
trainer.train()
print("===== LoRA 微调训练完成 =====")

# -------------------------- 2.4 恶意交易推理函数 --------------------------
def detect_malicious(opcode_seq):
    """输入操作码序列，输出交易检测结果"""
    text = opcode_to_text(opcode_seq)
    prompt = f"""任务：判断以太坊交易是否为恶意交易。
交易描述：{text}
请输出结果：0(正常) 或 1(恶意)
答案："""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = lora_model.generate(**inputs, max_new_tokens=2, temperature=0.1)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res[-1]

# 单条测试
test_op_seq = ["PUSH1", "CALL", "JUMP1", "SLOAD"]
detect_result = detect_malicious(test_op_seq)
print(f"\n[单条推理测试]")
print(f"交易操作码: {test_op_seq}")
print(f"检测结果(0=正常,1=恶意): {detect_result}")

# ========================== 端到端全流程：业务识别 + 恶意检测（论文完整系统） ==========================
def full_detection_pipeline(nx_call_graph, opcode_seq):
    """
    AIB2PSec 完整业务流程：
    1. 图网络识别项目业务类型
    2. 大模型检测交易是否恶意
    3. 输出告警结果
    """
    # 步骤1：业务相似度识别
    pyg_data = graph_to_pyg_data(nx_call_graph).to(device)
    pyg_data.batch = torch.zeros(pyg_data.x.shape[0], dtype=torch.long).to(device)
    gnn_out = gnn_model(pyg_data)
    business_type = gnn_out.argmax(dim=1).item()
    business_name = "DEX去中心化交易所项目" if business_type == 0 else "借贷类DeFi项目"

    # 步骤2：恶意交易检测
    mal_result = detect_malicious(opcode_seq)
    alert_msg = "⚠️ 告警：检测到恶意交易！已同步预警同类业务项目" if mal_result == "1" else "✅ 正常交易，无风险"

    # 输出结果
    print("\n===== AIB2PSec 智能合约安全检测系统 全流程结果 =====")
    print(f"项目业务类型: {business_name}")
    print(f"交易状态: {alert_msg}")
    return business_name, mal_result

# 执行端到端完整系统测试
if __name__ == "__main__":
    test_graph = generate_call_graph()
    test_op = ["CALL", "SWAP1", "JUMP1", "MSTORE", "POP"]
    full_detection_pipeline(test_graph, test_op)