# /--  秦东艺 --/
from extract_feature import BertVector
import json

import numpy as np
bv = BertVector()
with open(r'C:\Users\wyr\Desktop\知识点预测数据\魔方格教育网站相关\各学科知识点\魔方格物理知识点.txt', 'r', encoding = 'utf-8') as f:  # 打开文件
    data = f.readlines()  # 读取文件
    AllKPText = []
    for KP in data:
        KPText = KP.split(' ')[0]
        AllKPText.append(KPText) # 将全部知识点的名称存入一个list中
    EncoderList = bv.encode(AllKPText)

    KPWithFeature = {}
    i = 0
    for KP in AllKPText:
        KPWithFeature[KP] = EncoderList[i].tolist()
        i += 1


json_file = open('KPFeature_Physics.json', 'w', encoding='utf-8', errors='ignore')
json_str = json.dumps(KPWithFeature, indent = 4, ensure_ascii = False)
json_file.write(json_str)
json_file.close()