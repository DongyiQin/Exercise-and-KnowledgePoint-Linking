# !/usr/bin/python
# coding: utf8
# @Time    : 2020-11-8 19:09
# @Author  : DongyiQin
# @Email   : dongyics@gmail.com
# @Software: PyCharm
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import NetWork as net
from matplotlib import pyplot
import matplotlib.pyplot as plt


# 定义一些超参数
batch_size = 64
learning_rate = 0.01
num_epoches = 400


# 数据预处理。transforms.ToTensor()将文本转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize([0.5], [0.5])])


train_dataset = torch.load('./BV-PT/SRT_Q_KP_Label_Train1.pt')
test_dataset = torch.load('./BV-PT/SRT_Q_KP_Label_Test1.pt')
Validate_dataset = torch.load('./BV-PT/SRT_Q_KP_Label_Validate1.pt')


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)
validate_loader = DataLoader(Validate_dataset, batch_size = batch_size, shuffle = False)


# 选择模型
# model = net.simpleNet(1536, 768, 384, 2)
model = net.Activation_Net(1536, 768, 384, 1)
# model = net.Batch_Net(1536, 768, 384, 2)
optimizer = optim.SGD(model.parameters(), lr = learning_rate)


# # 训练模型
P_TO_P_Acc = [] # 记录验证集上的ACC，以便于
All_Loss = []
for epochs in range(num_epoches):
    p_to_p = 0
    p_to_n = 0
    n_to_n = 0
    n_to_p = 0

    VP_to_P_Last = 0
    VN_to_N_Last = 0
    Epoch_Loss_Sum = 0
    for data in train_loader:
        text, label = data
        text = Variable(text)
        label = Variable(label)


        out = model(text)

        P_EKP_Batch = []
        N_EKP_Batch = []
        Label_Batch = []
        group = out.reshape(-1, 8)

        P_EKP_Batch = group[:, 0].repeat(7).reshape(-1,7).t().reshape(-1,1)
        N_EKP_Batch = group[:, 1:].reshape(-1, 1)
        Target = torch.ones(P_EKP_Batch.shape[0]) # 目标：正例分数更高、反例分数更低
#
        loss = nn.MarginRankingLoss(margin = 0.3)
        loss = loss(P_EKP_Batch, N_EKP_Batch, Target)
        Epoch_Loss_Sum += loss # 累加一个epoch内的所有batch的loss

        optimizer.zero_grad() # grad在反向传播是累加的，反向传播前将梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        loc = 0
        for i in out:
            if i >= 0.5 and label[loc].equal(torch.tensor([1])): # 正例预测为正例的
                p_to_p += 1
            elif i >= 0.5 and label[loc].equal(torch.tensor([0])): # 负例预测为正例
                n_to_p += 1
            elif i < 0.5 and label[loc].equal(torch.tensor([0])): # 负例预测为负例的
                n_to_n += 1
            elif i < 0.5 and label[loc].equal(torch.tensor([1])): # 正例预测为负例
                p_to_n += 1
            loc +=  1

    Epoch_Loss = Epoch_Loss_Sum/len(train_loader)
    All_Loss.append(Epoch_Loss)
    print('epoch: {}, loss: {:.4}'.format(epochs+1, Epoch_Loss.data.item()))
    print('Train- P_TO_P_Acc: {:.6f}'.format(p_to_p / (p_to_p + p_to_n)))
    print('Train- N_TO_N_Acc: {:.6f}'.format(n_to_n / (n_to_n + n_to_p)))

    # 验证集上验证，防止过拟合
    Vp_to_p = 0
    Vp_to_n = 0
    Vn_to_n = 0
    Vn_to_p = 0
    for data_validate in validate_loader:
        text_validate, label_validate = data_validate
        text_validate = Variable(text_validate)
        label_validate = Variable(label_validate)

        out_validate = model(text_validate)
        loc_validate = 0
        for i in out_validate:
            if i >= 0.5 and label_validate[loc_validate].equal(torch.tensor([1])): # 正例预测为正例的
                Vp_to_p += 1
            elif i >= 0.5 and label_validate[loc_validate].equal(torch.tensor([0])): # 负例预测为正例
                Vn_to_p += 1
            elif i < 0.5 and label_validate[loc_validate].equal(torch.tensor([0])): # 负例预测为负例的
                Vn_to_n += 1
            elif i < 0.5 and label_validate[loc_validate].equal(torch.tensor([1])): # 正例预测为负例
                Vp_to_n += 1
            loc_validate += 1


    VP_to_P = Vp_to_p / (Vp_to_p + Vp_to_n)
    VN_to_N = Vn_to_n / (Vn_to_n + Vn_to_p)
    P_TO_P_Acc.append(VP_to_P)

    print('Validate- P_TO_P_Acc: {:.6f}'.format(VP_to_P))
    print('Validate- N_TO_N_Acc: {:.6f}'.format(VN_to_N))
    print('-------------------------------------------------')

    if len(P_TO_P_Acc) >10 and P_TO_P_Acc[-1] < P_TO_P_Acc[-2] and P_TO_P_Acc[-2] < P_TO_P_Acc[-3] and P_TO_P_Acc[-3] < P_TO_P_Acc[-4] :
        # 验证集上P_P结果连续4次都变差了，说明过拟合
        print('Avoid Overfitting！End Training！')
        break
    else:
        VP_to_P_Last = VP_to_P
        VN_to_N_Last = VN_to_N


# 模型评估
# model.eval()
Tp_to_p = 0
Tp_to_n = 0
Tn_to_n = 0
Tn_to_p = 0
Hits_at_One = 0
Hits_at_Three = 0
Hits_at_Five = 0
Sum = 0

Q_Loc = 0
for data_test in test_loader:
    Q_Loc += 1
    text_test, label_test = data_test
    text_test = Variable(text_test)
    label_test = Variable(label_test)

    out_test = model(text_test)
    group = out_test.reshape(-1, 16)

    P_EKP_Batch = group[:, 0].repeat(15).reshape(-1, 15).t().reshape(-1, 1)
    N_EKP_Batch = group[:, 1:].reshape(-1, 1)
    Target = torch.ones(P_EKP_Batch.shape[0])  # 目标：正例分数更高、反例分数更低

    loss = nn.MarginRankingLoss(margin = 0.3)
    loss = loss(P_EKP_Batch, N_EKP_Batch, Target)

    loc_test = 0
    for i in out_test:
        if i >= 0.7 and label_test[loc_test].equal(torch.tensor([1])): # 正例预测为正例的
            Tp_to_p += 1
        elif i >= 0.7 and label_test[loc_test].equal(torch.tensor([0])): # 负例预测为正例
            Tn_to_p += 1
        elif i < 0.3 and label_test[loc_test].equal(torch.tensor([0])): # 负例预测为负例的
            Tn_to_n += 1
        elif i < 0.3 and label_test[loc_test].equal(torch.tensor([1])): # 正例预测为负例
            Tp_to_n += 1
        loc_test +=  1

    Sum += len(group)
    for OneE in group:
        FourScore = []
        for OneScore in OneE:
            FourScore.append(OneScore.item())

        Sorted_FourScore = sorted(FourScore, reverse = True) # 分数从大到小排序
        if FourScore[0] in Sorted_FourScore[ : 1]: # 标注的知识点分数第一高
            Hits_at_One += 1
        if FourScore[0] in Sorted_FourScore[ : 3]: # 标注的知识点前三高
            Hits_at_Three += 1
        if FourScore[0] in Sorted_FourScore[ : 5]: # 标注的知识点前二高
            Hits_at_Five += 1

        # if FourScore[0] not in Sorted_FourScore[ : 3]:  # 标注的知识点不在前三里
        #     print(Q_Loc)



TP_to_P = Tp_to_p / (Tp_to_p + Tp_to_n)
TN_to_N = Tn_to_n / (Tn_to_n + Tn_to_p)
Hits_at_One = Hits_at_One / Sum
Hits_at_Three = Hits_at_Three / Sum
Hits_at_Five = Hits_at_Five / Sum
print('Test Loss: {:.6f}'.format(loss))
print('Test- P_TO_P_Acc: {:.6f}'.format(TP_to_P))
print('Test- N_TO_N_Acc: {:.6f}'.format(TN_to_N))
print('Hits@1: {:.6f}'.format(Hits_at_One))
print('Hits@3: {:.6f}'.format(Hits_at_Three))
print('Hits@5: {:.6f}'.format(Hits_at_Five))


# 绘制训练过程中的Loss曲线
X = range(len(All_Loss))
plt.plot(X, All_Loss, mec='r', mfc='w',label='Train Loss')
plt.title("TrainLoss", fontsize = 20)
plt.xlabel("Epoch", fontsize = 12)
plt.ylabel("Loss", fontsize = 12)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
plt.legend()
plt.show()