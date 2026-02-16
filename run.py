#模型使用接口（主观评估）

import torch
from net import Model
from transformers import BertTokenizer

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"F:\A\sentiment-binary-bert\model\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
model = Model().to(DEVICE)
names = ["负向评价","正向评价"]

#将传入的字符串进行编码
def collate_fn(data):
    sents = []
    sents.append(data)
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
    return input_ids,attention_mask,token_type_ids

def test():
    #加载模型训练参数
    model.load_state_dict(torch.load("params/best_bert.pth"))
    #开启测试模型
    model.eval()

    while True:
        data = input("请输入测试数据（输入‘q’退出）：")
        if data=='q':
            print("测试结束")
            break
        input_ids,attention_mask,token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE)

        #将数据输入到模型，得到输出
        with torch.no_grad():
            out = model(input_ids,attention_mask,token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：",names[out],"\n")

if __name__ == '__main__':
    test()