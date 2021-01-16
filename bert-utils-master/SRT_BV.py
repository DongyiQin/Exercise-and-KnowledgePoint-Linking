import torch
import xlrd
from extract_feature import BertVector
import numpy as np
import re
import json
import random # 使用python random模块的choice方法随机选择某个元素

def LoadJsonData(FilePath):
    with open(FilePath, 'r', encoding = 'utf-8') as Json:
        Json_dict = json.load(Json)
    return Json_dict


if __name__ == "__main__":
    bv = BertVector()
    PosJson_dict = LoadJsonData("DATA_EKP/正例.json")
    NegJson_dict = LoadJsonData("DATA_EKP/负采样.json")
    AllQ_KP_Label = []
    TEST_AllQ_KP_Label = []

    for Q in PosJson_dict:
        if len(NegJson_dict[Q]) >= 8: # 只取负例大于7的题目
            for Pos_KP in PosJson_dict[Q]: # 遍历所有的正例知识点
                Q_KP = []  # 题目，正例知识点，负例知识点1，负例知识点2，......,负例知识点7
                Q_KP.append(Q) # 题干
                Q_KP.append(Pos_KP) # 正例知识点
                OutOrder_NegKP = random.sample(NegJson_dict[Q], 8)   # 随机抽取8个负例
                num = 0
                for Neg_KP in OutOrder_NegKP:
                    if Neg_KP != Pos_KP:
                        num += 1
                        Q_KP.append(Neg_KP) # 负例知识点
                    if num==7:
                        break

            Q_KP_BV = bv.encode(Q_KP) # 题目-1正例知识点-7负例知识点 的编码

            QKP_BV_Label = [] # 将一个（题目-知识点向量，标签） 存入一个list中
            Q_KP_BV_CAT = torch.cat((torch.tensor(Q_KP_BV[0]), torch.tensor(Q_KP_BV[1])), dim=0) # 题干-正例知识点
            QKP_BV_Label.append(Q_KP_BV_CAT)  # 题目-知识点的串联bertvector
            QKP_BV_Label.append(torch.tensor([1]))  # 标签
            AllQ_KP_Label.append(QKP_BV_Label)


            for i in range(2,9): # 从负例知识点开始遍历
                Q_KP_BV_CAT = torch.cat((torch.tensor(Q_KP_BV[0]), torch.tensor(Q_KP_BV[i])), dim=0) # 题干-负例知识点
                QKP_BV_Label = []  # 将一个（题目-知识点向量，标签） 存入一个list中
                QKP_BV_Label.append(Q_KP_BV_CAT)  # 题目-知识点的串联bertvector
                QKP_BV_Label.append(torch.tensor([0]))  # 标签
                AllQ_KP_Label.append(QKP_BV_Label)



    for Q in PosJson_dict:
        if len(NegJson_dict[Q]) >= 16: # 只取负例大于15的题目.构造测试集
            for Pos_KP in PosJson_dict[Q]: # 遍历所有的正例知识点
                TEST_Q_KP = []  # 题目，正例知识点，负例知识点1，负例知识点2，......,负例知识点15
                TEST_Q_KP.append(Q) # 题干
                TEST_Q_KP.append(Pos_KP) # 正例知识点
                TEST_OutOrder_NegKP = random.sample(NegJson_dict[Q], 16)   # 随机抽取15个负例
                num = 0
                for Neg_KP in TEST_OutOrder_NegKP:
                    if Neg_KP != Pos_KP:
                        num += 1
                        TEST_Q_KP.append(Neg_KP) # 负例知识点
                    if num == 15:
                        break

            TEST_Q_KP_BV = bv.encode(TEST_Q_KP) # 题目-1正例知识点-15负例知识点 的编码

            TEST_QKP_BV_Label = [] # 将一个（题目-知识点向量，标签） 存入一个list中
            TEST_Q_KP_BV_CAT = torch.cat((torch.tensor(TEST_Q_KP_BV[0]), torch.tensor(TEST_Q_KP_BV[1])), dim=0) # 题干-正例知识点
            TEST_QKP_BV_Label.append(TEST_Q_KP_BV_CAT)  # 题目-知识点的串联bertvector
            TEST_QKP_BV_Label.append(torch.tensor([1]))  # 标签
            TEST_AllQ_KP_Label.append(TEST_QKP_BV_Label)


            for i in range(2,17): # 从负例知识点开始遍历
                TEST_Q_KP_BV_CAT = torch.cat((torch.tensor(TEST_Q_KP_BV[0]), torch.tensor(TEST_Q_KP_BV[i])), dim=0) # 题干-负例知识点
                TEST_QKP_BV_Label = []  # 将一个（题目-知识点向量，标签） 存入一个list中
                TEST_QKP_BV_Label.append(TEST_Q_KP_BV_CAT)  # 题目-知识点的串联bertvector
                TEST_QKP_BV_Label.append(torch.tensor([0]))  # 标签
                TEST_AllQ_KP_Label.append(TEST_QKP_BV_Label)

    # Len = len(AllQ_KP_Label)
    # print(Len) # 1880,按照9：1比例划分训练集和验证集
    # print(len(TEST_AllQ_KP_Label))
    Train = AllQ_KP_Label[ : 1688]
    Test = AllQ_KP_Label[1312 : 1688]
    Validate = AllQ_KP_Label[1688 : ]


    # Test = TEST_AllQ_KP_Label
    #
    #
    torch.save(Train, 'SRT_Q_KP_Label_Train2.pt')
    torch.save(Test, 'SRT_Q_KP_Label_Test2.pt')
    torch.save(Validate, 'SRT_Q_KP_Label_Validate2.pt')


    # Train = torch.load('SRT_Q_KP_Label_Train.pt')
    # Test = torch.load('SRT_Q_KP_Label_Test.pt')
    # Validate = torch.load('SRT_Q_KP_Label_Validate.pt')