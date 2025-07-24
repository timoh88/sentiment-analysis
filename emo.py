import requests
from snownlp import SnowNLP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

comment_url = ''
comments_list = []

for i in range(10):  # 爬取10页评论
    params = {
        'productId': 0,  # 商品ID
        'page': i,                 # 页码
        'pageSize': 10,             # 每页10条
        'sortType': 5               # 排序方式
    }
    headers = {...}  # 请求头(实际代码需补充具体内容)
    
    # 发送请求获取数据
    comment_resp = requests.get(comment_url, params=params, headers=headers)
    comment_dict = comment_resp.json()
    
    # 解析评论内容
    for comment in comment_dict['comments']:
        # 处理换行分割的评论
        temp_comments = comment['content'].split('\n')
        for temp_comment in temp_comments:
            comments_list.append(temp_comment)

            emotions = []
for comment in comments_list:
    try:
        # 使用SnowNLP进行情感分析（0-1之间，>0.5为积极）
        s = SnowNLP(comment)
        emotions.append(s.sentiments)
    except ZeroDivisionError:  # 处理空文本异常
        emotions.append(None)

# 创建数据框
df = pd.DataFrame({
    'comments': comments_list,
    'emotion': emotions
})

# 查看情感分数统计
print(df.describe())

pos, neg = 0, 0
for score in df['emotion']:
    if score >= 0.5:
        pos += 1  # 积极计数
    else:
        neg += 1  # 消极计数

print(f'积极评论: {pos}\n消极评论: {neg}')

pie_labels = ['positive', 'negative']
plt.pie([pos, neg], 
        labels=pie_labels,
        autopct='%1.2f%%',  # 显示百分比
        shadow=True)
plt.savefig('积极消极评论占比.png')

plt.clf()  # 清除上一图形
bins = np.arange(0, 1.1, 0.1)  # 创建0-1的10个区间

plt.hist(df['emotion'], bins,
         color='#4F94CD',  # 设置蓝色
         alpha=0.9)       # 透明度

plt.xlim(0, 1)            # X轴范围
plt.xlabel('情感分')       # X轴标签
plt.ylabel('数量')         # Y轴标签
plt.title('情感分直方图')  # 标题
plt.savefig('情感分直方图.png')