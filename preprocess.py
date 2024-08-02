import pandas as pd
from bs4 import BeautifulSoup
import re

# 读取CSV文件
df = pd.read_csv('./yf_data_develop.csv')

# 删除指定的字段
columns_to_remove = ['id', 'domain', 'createAt', 'imageUrl',
                      'language', 'is_insert_es']
df = df.drop(columns=columns_to_remove)

# 定义一个函数来清理content字段中的HTML和URL
def clean_content(html_content):
    if isinstance(html_content, str):
        # 使用BeautifulSoup去除HTML标签
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()

        # 使用正则表达式去除URL
        text = re.sub(r'http\S+', '', text)  # 去除http开头的url
        text = re.sub(r'www\S+', '', text)   # 去除www开头的url

        return text.strip()
    else:
        return ""  # 如果content不是字符串，返回空字符串


# 清理content字段
df['content'] = df['content'].apply(clean_content)

# 保存清洗后的数据到新的CSV文件
df.to_csv('cleaned_output.csv', index=False)

print("Data cleaning completed and saved to cleaned_output.csv")




# id	domain	emotion	createAt	imageUrl	language	nickName	readCount	title	url	userName	websiteName	content	is_insert_es
