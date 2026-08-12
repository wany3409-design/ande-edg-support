#!/usr/bin/env python
"""
Embedding 模型加载与测试脚本
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("  Text2Vec Embedding 模型测试")
print("=" * 60)

# Step 1: 下载并加载模型
print("\n[1/4] 加载模型 shibing624/text2vec-base-chinese ...")
t0 = time.time()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("shibing624/text2vec-base-chinese", device="cpu")

t1 = time.time()
print(f"  模型加载完成 (耗时 {t1 - t0:.1f}s)")

# Step 2: 获取模型信息
print("\n[2/4] 模型信息:")
print(f"  模型名称: shibing624/text2vec-base-chinese")
print(f"  向量维度: {model.get_sentence_embedding_dimension()}")
print(f"  最大序列长度: {model.max_seq_length}")

# Step 3: 最小Embedding测试
print("\n[3/4] Embedding 测试...")
test_texts = [
    "文件加密策略配置",
    "客户端连接服务端端口",
    "AD域同步配置方法",
    "例外目录如何设置",
    "文件外发审批流程",
]

t2 = time.time()
embeddings = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)
t3 = time.time()

print(f"  测试文本数: {len(test_texts)}")
print(f"  向量形状: {embeddings.shape}")
print(f"  总耗时: {t3 - t2:.1f}s")
print(f"  单条耗时: {(t3 - t2) / len(test_texts) * 1000:.1f}ms")

# 验证语义相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print(f"\n  语义相似度矩阵:")
for i, text in enumerate(test_texts):
    top_idx = sorted(range(len(test_texts)), key=lambda j: sim_matrix[i][j], reverse=True)[1]
    print(f"    '{text[:20]}...' -> 最相似: '{test_texts[top_idx][:20]}...' ({sim_matrix[i][top_idx]:.3f})")

# Step 4: 磁盘占用
print("\n[4/4] 磁盘占用:")
import subprocess
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_dir = None
for root, dirs, files in os.walk(cache_dir):
    if "text2vec-base-chinese" in root:
        model_dir = root
        break

if model_dir:
    du_output = subprocess.run(
        ["du", "-sh", model_dir],
        capture_output=True, text=True, shell=True
    )
    print(f"  模型缓存目录: {model_dir}")
    print(f"  占用: {du_output.stdout.strip() if du_output.stdout else 'N/A'}")
else:
    print(f"  模型目录未在 {cache_dir} 下找到，可能在 HuggingFace 缓存中")

# C盘剩余空间
import shutil
usage = shutil.disk_usage("C:/")
free_gb = usage.free / (1024**3)
total_gb = usage.total / (1024**3)
print(f"\n  C盘总量: {total_gb:.1f} GB")
print(f"  C盘剩余: {free_gb:.1f} GB")

print("\n" + "=" * 60)
print("  模型加载测试通过!")
print("=" * 60)
