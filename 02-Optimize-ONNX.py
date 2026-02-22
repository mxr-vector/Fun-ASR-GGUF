import os
from pathlib import Path
import onnx
from onnxruntime.transformers.optimizer import optimize_model
from collections import defaultdict
from export_config import EXPORT_DIR

def optimize_and_check(input_path, output_path):
    print(f"\n===========================")
    print(f"正在优化模型: {input_path}")
    print(f"===========================\\n")
    
    # 使用 bert 模型类型可以触发最多的常规 Transformer 融合（如 Gelu、LayerNorm、Attention）
    optimizer = optimize_model(
        input_path,
        model_type='bert',          
        num_heads=0,               # 设为0代表不强制检查自注意力头的数量匹配
        hidden_size=0,
        opt_level=1,               # 基本算子融合
    )
    
    optimizer.save_model_to_file(output_path)
    print(f"✅ 模型优化已完成，另存为: {output_path}\n")

    print(f"--- 内部细节诊断 ---")
    model = onnx.load(output_path, load_external_data=False)
    
    print("导入的 Opset Versions:")
    for imp in model.opset_import:
        domain = imp.domain if imp.domain else 'ai.onnx (官方标准)'
        print(f"  - Domain: {domain}, Version: {imp.version}")
        
    domain_ops = defaultdict(set)
    for node in model.graph.node:
        domain = node.domain if node.domain else 'ai.onnx'
        domain_ops[domain].add(node.op_type)
        
    print("\n优化后图内包含的算子分布:")
    for domain, ops in sorted(domain_ops.items()):
        print(f"\n[{domain}]")
        ops_list = sorted(list(ops))
        for i in range(0, len(ops_list), 5):
            print("  " + ", ".join(ops_list[i:i+5]))

def main():
    output_dir = Path(EXPORT_DIR)
    
    targets = [
        output_dir / "Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
        output_dir / "Fun-ASR-Nano-CTC.fp32.onnx"
    ]
    
    for path in targets:
        if path.exists():
            optimize_and_check(str(path), str(path))
        else:
            print(f"❌ 找不到输入模型: {path}")

if __name__ == "__main__":
    main()
