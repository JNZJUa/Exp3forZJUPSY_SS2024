# README.md
# By Jin Nan. 20240418
本项目为浙江大学《信号与认知系统》2024春学期课程，作业3代码与数据集。

**使用前说明：**
- 运行前请确保环境中安装了所有必须的库（例如pytorch）。
- CBOW模型的代码为`CBOW_V5.py`。
- Skip-gram模型的代码为`SKIPgram_V1.py`。
- 代码同目录下需已有GloVe100维词向量。

## CBOW_V5.py说明：
- 使用GLoVe作为初始化嵌入层，因此请确保代码同目录下已有GloVe100维词向量。
- 该模型主要用于预测以下任务：
  - 使用训练好的CBOW模型预测下列句子中空白处最可能的单词以及非常不可能的单词：
    - "I ate a _____ for breakfast."
    - "I saw a huge _____ yesterday."
- 用到的数据分为训练数据集`text_train.txt`和验证数据集`text_test.txt`，以及`text_train.txt`与`text_test.txt`内容简单拼接的`text_all.txt`。使用时请将这三个文件里的文本数据替换为自己的文本。
  - 同时特别提醒，为完成上述预测任务，训练数据集`text_train.txt`中应包括需预测的两个句子原句。
- 代码输出：
  - `CBOW_V5.py`将输出一个Excel表格，记录了每轮试次中模型对训练数据集、验证数据的平均损失值；同时记录了每轮训练结束后CBOW模型在预测任务的上的预测结果，以便观测从第几轮训练后，模型可准确预测；同时`time_cost`记录了训练总耗时。
  - `CBOW_V5.py`还将输出使用t-SNE降为2维的词向量示意图。

## SKIPgram_V1.py说明：
- 基本同CBOW_V5.py一致。
- 但主要用于预测以下任务：
  - 使用训练好的Skip-gram模型预测中心词“croissant”“cloud”的上下文（这两个词也就是CBOW预测任务的两个缺失词）。
- 代码输出：
  - 同CBOW_V5.py一致。
