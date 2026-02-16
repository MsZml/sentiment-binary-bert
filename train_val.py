#模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer,AdamW
import os

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次(将整个数据集训练完一次为一轮)
EPOCH = 30000

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"F:\A\sentiment-binary-bert\model\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids,attention_mask,token_type_ids,label



#创建数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    #训练批次
    batch_size=50,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载的数据进行编码
    collate_fn=collate_fn
)
#创建验证数据集
val_dataset = MyDataset("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    #训练批次
    batch_size=50,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载的数据进行编码
    collate_fn=collate_fn
)
if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamW(model.parameters())
    #定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    #初始化验证最佳准确率
    best_val_acc = 0.0

    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(train_loader):
            #将数据放到DVEVICE上面
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),label.to(DEVICE)
            #前向计算（将数据输入模型得到输出）
            out = model(input_ids,attention_mask,token_type_ids)
            #根据输出计算损失
            loss = loss_func(out,label)
            #根据误差优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5 ==0:
                out = out.argmax(dim=1)
                #计算训练精度
                acc = (out==label).sum().item()/len(label)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        #验证模型（判断模型是否过拟合）
        #设置为评估模型
        model.eval()
        total_correct = 0
        total_samples = 0
        val_loss = 0.0

        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(val_loader):
                input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(
                    DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
                out = model(input_ids, attention_mask, token_type_ids)

                # 计算批次损失（已取平均）
                loss_batch = loss_func(out, label)
                val_loss += loss_batch.item()

                # 统计正确预测数和总样本数
                preds = out.argmax(dim=1)
                total_correct += (preds == label).sum().item()
                total_samples += label.size(0)  # 当前批次样本数

            # 计算整个验证集的平均损失和准确率
            val_loss /= len(val_loader)
            val_acc = total_correct / total_samples  # 使用总样本数
            print(f"验证集：loss:{val_loss},acc:{val_acc}")
        # #每训练完一轮，保存一次参数
        # torch.save(model.state_dict(),f"params/{epoch}_bert.pth")
        # print(epoch,"参数保存成功！")
            # ===== 自动创建params目录如果 =====
            save_dir = "params"
            # 检查目录是否存在，不存在则创建
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # 递归创建目录（即使父目录不存在也能创建）
            #  =====
            #根据验证准确率保存最优参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),"params/best_bert.pth")
                print(f"EPOCH:{epoch}:保存最优参数：acc{best_val_acc}")
        #保存最后一轮参数
        torch.save(model.state_dict(), "params/last_bert.pth")
        print(f"EPOCH:{epoch}:最后一轮参数保存成功！")

