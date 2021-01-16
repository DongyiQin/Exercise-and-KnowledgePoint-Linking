import pandas as pd
import jieba
import xlrd
import xlsxwriter
import json


def GetDepart(Text):
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='gbk').readlines()]  # 创建停用词列表
    Text_depart_List = []
    for i in Text:
        Text_depart = jieba.cut(i)
        outstr = ''
        # 去停用词
        for word in Text_depart:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        Text_depart_List.append(outstr)
    return  Text_depart_List


def GetDepartList(Corpus):
    ResultList = []
    for i in Corpus:
        Result = i.split()
        ResultList.append(Result)
    return ResultList


ExcelFile = r"尚瑞通手动标注数据.xlsx"
df = pd.read_excel(ExcelFile, sheet_name='Sheet1', header=None)
QuestionText = df[0]
Q_Departed_List = GetDepart(QuestionText)
# print(Q_Departed_List)


AllKP = []
#读取excel文件
Excelfile=xlrd.open_workbook(r'数学学科KG.xlsx')
sheet = Excelfile.sheet_by_index(0)
#获取行内容，索引从0开始
nrows = sheet.nrows      # 获取表的行数
for i in range(1, nrows):   # 循环逐行打印
    for j in range(6, 8): # 取第五层知识点
        if sheet.row_values(i)[j] and sheet.row_values(i)[j] not in AllKP:
            AllKP.append(sheet.row_values(i)[j])
AllKP_Departed_List = GetDepart(AllKP)
# print(AllKP_Departed_List)


Q_Departed_List = GetDepartList(Q_Departed_List)
AllKP_Departed_List = GetDepartList(AllKP_Departed_List)
# print(Q_Departed_List)
# print(AllKP_Departed_List)


AllNegSample = {}
Qloc = 0
for Q in Q_Departed_List:
    OneNegSample = []
    loc = 0
    # print(QuestionText[Qloc])
    for KP in AllKP_Departed_List:
        for word in KP:
            if word in Q and AllKP[loc] and AllKP[loc] not in OneNegSample:
                OneNegSample.append(AllKP[loc])
        loc += 1
    AllNegSample[QuestionText[Qloc]] = OneNegSample[0:20]
    Qloc += 1


json_file = open('负采样.json', 'w', encoding='utf-8', errors='ignore')
json_str = json.dumps(AllNegSample, indent = 4, ensure_ascii = False)
json_file.write(json_str)
json_file.close()

#
# AllOppSample = {}
# Excelfile = xlrd.open_workbook(r'尚瑞通手动标注数据.xlsx')
# sheet = Excelfile.sheet_by_index(0)
# #获取行内容，索引从0开始
# nrows = sheet.nrows      # 获取表的行数
# for i in range(1, nrows):   # 循环逐行打印
#     AllOppSample[sheet.row_values(i)[0]] = []
#     for j in range(1, 6): # 取第五层知识点
#         if sheet.row_values(i)[j]:
#             AllOppSample[sheet.row_values(i)[0]].append(sheet.row_values(i)[j])
# print(AllOppSample)
#
# json_file = open('正例.json', 'w', encoding='utf-8', errors='ignore')
# json_str = json.dumps(AllOppSample, indent = 4, ensure_ascii = False)
# json_file.write(json_str)
# json_file.close()