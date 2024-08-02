import pandas as pd
import numpy as np
from xinference.client import Client
import json

# 模型服务端点和初始化
client = Client("http://127.0.0.1:9997")
model = client.get_model('bge-reranker-v2-m3')

# 读取CSV文件
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data['content']


def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.astype(str)  # 将文本列转换为纯字符串
    return data

# 使用rerank模型进行排序
def rerank_data(data, query):
    corpus = data['content'].tolist()[:200]
    
    # 使用模型进行重新排序
    ranked_indices = model.rerank(corpus, query)
    
    # 根据返回的排序索引重新排列数据
    
    return ranked_indices

def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON data saved to {output_file_path}")

def main(input_file_path, output_file_path, query):
    data = read_csv(input_file_path)
    clean_data(data)  # 清理数据中的异常值
    ranked_data = rerank_data(data, query)
    save_to_json(ranked_data, output_file_path)
    print(f"Re-ranked data saved to {output_file_path}")

# 设置文件路径和查询
input_file_path = "./yf_data_develop.csv"
output_file_path = "./output.json"

query = '''
'''  # 替换为实际的查询内容

# 执行程序
# main(input_file_path, output_file_path, query)

content = read_csv(input_file_path)
corpus = content.astype(str).tolist()
# print(corpus[:20])
ranked_indices = model.rerank(corpus[:122], query)
save_to_json(ranked_indices, output_file_path)
# print(ranked_indices)







