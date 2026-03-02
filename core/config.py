import argparse
import torch
import os
import yaml

def __build_parser_args() -> argparse.Namespace:
    """
    构建参数解析器
    """
    parser = argparse.ArgumentParser(description="Fun ASR GGUF Configuration")
    
    # 通用参数
    parser.add_argument("--use_gpu", type=bool, default=torch.cuda.is_available(), help="是否使用GPU预测")
    parser.add_argument("--web_secret_key", type=str, default="funasr-nano-token", help="接口请求秘钥")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务启动IP地址")
    parser.add_argument("--port", type=int, default=8001, help="服务启动端口")
    parser.add_argument("--base_url", type=str, default="/funasr/api/v1", help="接口基础路径")
    parser.add_argument("--configs", type=str, default="", help="配置文件路径")

    # 使用 parse_known_args 防止未知参数报错
    args, _ = parser.parse_known_args()
    return args

args = __build_parser_args()
