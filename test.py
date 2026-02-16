# test.py - 模型评估测试模块
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载字典和分词器
token = BertTokenizer.from_pretrained(
    r"F:\A\sentiment-binary-bert\model\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")


# 将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    # 编码
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
    return input_ids, attention_mask, token_type_ids, label


def evaluate_model(model, test_loader, device):
    """
    评估模型在测试集上的性能
    :param model: 待评估模型
    :param test_loader: 测试数据加载器
    :param device: 计算设备
    :return: 评估指标字典
    """
    model.eval()
    all_preds, all_labels = [], []

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据转移到设备
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)

        # 收集预测结果
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted'),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, digits=4)
    }
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制并保存混淆矩阵
    :param cm: 混淆矩阵
    :param class_names: 类别名称列表
    :param save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('yucebiaoqian')
    plt.ylabel('zhenshibiaoqian')
    plt.title('hunxiaojuzhen')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    plt.show()


def save_metrics_to_file(metrics, save_path):
    """
    将评估指标保存到文本文件
    :param metrics: 评估指标字典
    :param save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("模型评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n\n")

        f.write("宏平均指标 (Macro-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_macro']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_macro']:.4f}\n\n")

        f.write("加权平均指标 (Weighted-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_weighted']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_weighted']:.4f}\n\n")

        f.write("分类报告 (Classification Report):\n")
        f.write(metrics['classification_report'])

        f.write("\n\n混淆矩阵 (Confusion Matrix):\n")
        np.savetxt(f, metrics['confusion_matrix'], fmt='%d')

    print(f"评估报告已保存至: {save_path}")


if __name__ == '__main__':
    # 创建数据集
    test_dataset = MyDataset("test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,  # 评估时不需要打乱
        drop_last=False,  # 保留所有样本
        collate_fn=collate_fn
    )

    # 开始测试
    print(f"使用设备: {DEVICE}")
    model = Model().to(DEVICE)

    # 模型参数路径
    model_path = "params/best_bert.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型参数文件不存在: {model_path}")

    # 加载模型训练参数
    model.load_state_dict(torch.load(model_path))

    # 评估模型
    metrics = evaluate_model(model, test_loader, DEVICE)

    # 打印评估结果
    print("\n" + "=" * 50)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print("\n宏平均指标 (Macro-average):")
    print(f"  精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"  F1分数 (F1 Score): {metrics['f1_macro']:.4f}")

    print("\n加权平均指标 (Weighted-average):")
    print(f"  精确率 (Precision): {metrics['precision_weighted']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_weighted']:.4f}")
    print(f"  F1分数 (F1 Score): {metrics['f1_weighted']:.4f}")

    print("\n分类报告 (Classification Report):")
    print(metrics['classification_report'])

    # 可视化混淆矩阵
    # 注意：根据您的实际类别修改class_names
    class_names = ["label0", "label1"]  # 替换为您的实际类别名称
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, "confusion_matrix.png")

    # 保存评估结果
    save_metrics_to_file(metrics, "evaluation_report.txt")

    print("评估完成!")