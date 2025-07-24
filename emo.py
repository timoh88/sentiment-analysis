import requests
from snownlp import SnowNLP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ��������������ʾ
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

comment_url = ''
comments_list = []

for i in range(10):  # ��ȡ10ҳ����
    params = {
        'productId': 0,  # ��ƷID
        'page': i,                 # ҳ��
        'pageSize': 10,             # ÿҳ10��
        'sortType': 5               # ����ʽ
    }
    headers = {...}  # ����ͷ(ʵ�ʴ����貹���������)
    
    # ���������ȡ����
    comment_resp = requests.get(comment_url, params=params, headers=headers)
    comment_dict = comment_resp.json()
    
    # ������������
    for comment in comment_dict['comments']:
        # �����зָ������
        temp_comments = comment['content'].split('\n')
        for temp_comment in temp_comments:
            comments_list.append(temp_comment)

            emotions = []
for comment in comments_list:
    try:
        # ʹ��SnowNLP������з�����0-1֮�䣬>0.5Ϊ������
        s = SnowNLP(comment)
        emotions.append(s.sentiments)
    except ZeroDivisionError:  # ������ı��쳣
        emotions.append(None)

# �������ݿ�
df = pd.DataFrame({
    'comments': comments_list,
    'emotion': emotions
})

# �鿴��з���ͳ��
print(df.describe())

pos, neg = 0, 0
for score in df['emotion']:
    if score >= 0.5:
        pos += 1  # ��������
    else:
        neg += 1  # ��������

print(f'��������: {pos}\n��������: {neg}')

pie_labels = ['positive', 'negative']
plt.pie([pos, neg], 
        labels=pie_labels,
        autopct='%1.2f%%',  # ��ʾ�ٷֱ�
        shadow=True)
plt.savefig('������������ռ��.png')

plt.clf()  # �����һͼ��
bins = np.arange(0, 1.1, 0.1)  # ����0-1��10������

plt.hist(df['emotion'], bins,
         color='#4F94CD',  # ������ɫ
         alpha=0.9)       # ͸����

plt.xlim(0, 1)            # X�᷶Χ
plt.xlabel('��з�')       # X���ǩ
plt.ylabel('����')         # Y���ǩ
plt.title('��з�ֱ��ͼ')  # ����
plt.savefig('��з�ֱ��ͼ.png')