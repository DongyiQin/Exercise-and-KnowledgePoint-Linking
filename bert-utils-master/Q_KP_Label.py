import torch
import xlrd
from extract_feature import BertVector
import numpy as np
import re
if __name__ == "__main__":
    bv = BertVector()
    Excelfile = xlrd.open_workbook(r'E_KP_DATA.xlsx')
    table = Excelfile.sheets()[1]
    # print(table)
    nrows = table.nrows  # 获取表的行数
    AllQ_KP_Label = []


    for i in range(1, nrows):  # 循环逐行打印,跳过第一行表头
        OneLine = table.row_values(i)
        # print(OneLine)
        # ['下面四个数中，最小的数是（）', '有理数定义及分类', '1']

        # Q_KP = []
        # Q_KP.append(OneLine[0])
        # Q_KP.append(OneLine[1])
        BertVector = bv.encode(Q_KP)  # 只编码题干和知识点
        # print(BertVector)
        Q_KP_BertVector_CAT = torch.cat((torch.tensor(BertVector[0]), torch.tensor(BertVector[1])), dim=0)  # 串联题目和知识点的vector
        # print(Q_KP_BertVector_CAT) # 串联后bertvector长度 768*2 = 1536

        Label = OneLine[2]
        if Label == '0':
            Label_Tensor = torch.tensor([0])
        else:
            Label_Tensor = torch.tensor([1])
        # print(Label_Tensor.shape)
        # break
        QKP_BV_Label = []  # 将一个（题目-知识点向量，标签） 存入一个list中
        QKP_BV_Label.append(Q_KP_BertVector_CAT)  # 题目-知识点的串联bertvector
        QKP_BV_Label.append(Label_Tensor) # 标签

        AllQ_KP_Label.append(QKP_BV_Label)

    torch.save(AllQ_KP_Label, 'Q_KP_Label_Test.pt')
    # xy_list = torch.load('Q_KP_Label.pt')
    # print(xy_list)

