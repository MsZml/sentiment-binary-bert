<div align="center">
  
# 基于BERT-base-chinese实现中文情感二分类
</div>

## 本项目基于 BERT-base-chinese 预训练模型实现中文文本情感二分类任务（正向 / 负向评价），包含完整的训练、验证、测试流程，以及交互式预测接口，可直接用于中文评论、商品评价等场景的情感分析。

### 项目目录：
├── data                    # 数据集，包括训练集和验证集以及测试集  
├── model                   # bert-base-chinese模型  
├── MyData.py               # 自定义数据集加载类（基于datasets库）  
├── net.py                  # BERT增量模型定义（二分类下游任务）  
├── train_val.py            # 模型训练&验证主程序  
├── test.py                 # 模型评估测试（含混淆矩阵、分类报告）  
├── run.py                  # 交互式预测接口（主观测试）  
├── token_test.py           # BERT分词器测试示例 
├── data_test.py            # 数据集加载测试示例  
├── params/                 # 模型参数保存目录（自动生成）   
│   └── best_bert.pth       # 训练完成后的最优模型参数  
│   └── last_bert.pth       # 训练完成后的最后模型参数   
├── confusion_matrix.png    # 测试集混淆矩阵可视化结果（运行test.py生成）  
├── evaluation_report.txt   # 测试集评估报告（运行test.py生成）  
└── README.md               # 项目说明文档  

### 环境：

本次使用python=3.10

cuda 12.6

PyTorch 2.5.1+cu124

### 依赖：

>pip install torch transformers datasets scikit-learn matplotlib seaborn numpy

（版本可能差异，如果有bug自己调整一下）

### 数据准备；

数据集来源，本项目使用 ChnSentiCorp 中文情感分析数据集

### 数据集路径配置：

修改 MyData.py 中以下路径为你的数据集实际路径：

`self.dataset = load_from_disk(r"F:\A\sentiment-binary-bert\data\ChnSentiCorp")`

### 模型：

使用 Hugging Face 的 transformers 库加载 bert-base-chinese 预训练模型，下载好放在在model文件下

#### 修改以下文件中的模型路径为你的实际路径；
net.py

run.py

test.py

token_test.py

### 模型说明：
#### 预训练模型：
BERT-base-chinese（12 层 Transformer，768 维隐藏层）

#### 下游任务：
二分类（全连接层将 768 维特征映射为 2 维输出）

#### 训练策略：

冻结 BERT 预训练参数，仅训练全连接层（增量学习）

### 训练模型：

#### 运行 train_val.py 启动训练，默认配置：

训练轮数（EPOCHS）：5

批次大小（BATCH_SIZE）：50

学习率（LEARNING_RATE）：1e-3

设备：自动适配 GPU/CPU

### 测试模型：

#### 运行 test.py 评估模型在测试集上的性能，自动生成：

混淆矩阵可视化图（confusion_matrix.png）

详细评估报告（evaluation_report.txt）

### 交互式预测：

运行 run.py 启动交互式测试，输入任意中文文本即可得到情感分类结果

### 注意：先下好数据集和模型，训练批次理想是30000，但是自己电脑大概率跑不了那么久，可以改小一点，大概5轮左右就将近90%了

##本项目仅供学习交流使用
![Uploading image.png…]()
