import torch
import torch.nn as nn
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_glove_embeddings(path, word_to_ix, embedding_dim):
    # 加载GloVe词向量
    """
    在这段代码中，首先初始化了一个随机的嵌入矩阵 embeddings，每个词向量的维度由embedding_dim指定。
    随后，从GloVe的预训练词向量文件中读取词向量，如果词汇在字典word_to_ix中存在，则用对应的GloVe向量替换原来随机初始化的向量。
    """
    embeddings = np.random.rand(len(word_to_ix), embedding_dim)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_ix:
                #print(f"Processing word......\n")
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word_to_ix[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float)

def make_context_vector(context, word_to_ix):
    # 将文本上下文转换为索引向量
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def load_text_modified(file_path):
    # Read and lower-case the entire text
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()

    # Split text into sentences
    #sentences = re.split(r"[.!?]", text)
    sentences = re.split(r'[.!?;"]+', text)
    all_words = []

    for sentence in sentences:
        # Remove non-alphabet characters and split into words
        cleaned_sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)
        words = cleaned_sentence.split()
        if words:  # Ensure the sentence is not empty
            all_words.extend(words)  # Append words to the list

    # Add "<start>" at the beginning and "<end>" at the end of the list
    processed_text = ["<start>"] + all_words + ["<end>"]
    return processed_text

# Commenting out function call for the purpose of preventing execution here
# load_text_modified("path_to_your_text_file.txt")


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()  # 先转换为小写

    # 使用正则表达式划分句子
    #sentences = re.split(r"[.!?]", text)  # 根据句号、问号和感叹号分割句子
    sentences = re.split(r'[.!?;"]+', text)
    processed_sentences = []

    for sentence in sentences:
        # 去除每句话中的非字母字符
        sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)
        words = sentence.split()
        if words:  # 确保句子不为空
            processed_sentences.extend(["<start>"] + words + ["<end>"])

    return processed_sentences



# 常量定义
CONTEXT_SIZE = 2  # 左右各2个单词
EMBEDDING_DIM = 100
GLOVE_PATH = 'glove.6B.100d.txt'
TEXT_PATH = 'text_All.txt'

# 加载和处理文本-整个数据集
raw_text = load_text(TEXT_PATH)
vocab = set(raw_text)
vocab_size = len(vocab)
word_to_ix = {word: ix for ix, word in enumerate(vocab)}
ix_to_word = {ix: word for ix, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

# 加载和处理文本 验证数据集
TEXT_PATH_test = 'text_test.txt'
raw_text_test = load_text(TEXT_PATH_test)
data_test = []
for i in range(2, len(raw_text_test) - 2):
    context = [raw_text_test[i - 2], raw_text_test[i - 1], raw_text_test[i + 1], raw_text_test[i + 2]]
    target = raw_text_test[i]
    data_test.append((context, target))

# 加载和处理文本 验证数据集
TEXT_PATH_train = 'text_train.txt'
raw_text_train = load_text(TEXT_PATH_train)
data_train = []
for i in range(2, len(raw_text_train) - 2):
    context = [raw_text_train[i - 2], raw_text_train[i - 1], raw_text_train[i + 1], raw_text_train[i + 2]]
    target = raw_text_train[i]
    data_train.append((context, target))

# 模型定义
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(CBOW, self).__init__()
        # 使用预训练的词向量初始化嵌入层
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        # 前向传播过程
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

# 加载词向量并创建模型
pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word_to_ix, EMBEDDING_DIM)
model = CBOW(vocab_size, EMBEDDING_DIM, pretrained_embeddings)



def predict_missing_word(sentence, word_to_ix, ix_to_word, model):
    # 对句子进行小写转换并去除非字母字符，同时保留“_____”占位符
    sentence = re.sub(r"[^\w\s_]", "", sentence).lower()  # 仅去除非单词字符、空格或下划线
    words = ['<start>', '<start>'] + sentence.split() + ['<end>', '<end>']  # 添加开始和结束标记
    try:
        blank_index = words.index("_____")
    except ValueError:
        return "Error: Placeholder '_____' not found in the sentence."

    # 确定上下文
    context = words[max(0, blank_index - 2):blank_index] + words[blank_index + 1:blank_index + 3]
    context_vector = make_context_vector(context, word_to_ix)

    # 使用模型进行预测
    log_probs = model(context_vector)
    probabilities = torch.exp(log_probs)

    # 寻找最可能和最不可能的单词
    _, idxs = torch.sort(probabilities, descending=True)
    most_likely_word = ix_to_word[idxs[0, 0].item()] if idxs.size(1) > 0 else 'No prediction'
    least_likely_word = ix_to_word[idxs[0, -1].item()] if idxs.size(1) > 0 else 'No prediction'

    return most_likely_word, least_likely_word

# 训练设置
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 数据集分为训练集和验证集
training_data = data_train
validation_data = data_test

# 用于记录损失值和预测结果的列表
train_losses = []
validation_losses = []
predictions_per_epoch = []
import time

# 在训练之前记录开始时间
start_time = time.time()

for epoch in range(100):
    total_train_loss = 0
    model.train()  # 设置模型为训练模式
    for context, target in training_data:
        context_vector = make_context_vector(context, word_to_ix)
        log_probs = model(context_vector)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_losses.append(total_train_loss / len(training_data))
    #train_losses.append(total_train_loss)

    # 验证过程
    total_val_loss = 0
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在计算验证损失时不需要计算梯度
        for context, target in validation_data:
            context_vector = make_context_vector(context, word_to_ix)
            log_probs = model(context_vector)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            total_val_loss += loss.item()
    validation_losses.append(total_val_loss / len(validation_data))
    #validation_losses.append(total_val_loss)

    # 验证结束后，保存预测结果
    epoch_predictions = []
    sentences = ["I ate a _____ for breakfast", "I saw a huge _____ yesterday"]
    for sentence in sentences:
        most_likely, least_likely = predict_missing_word(sentence, word_to_ix, ix_to_word, model)
        epoch_predictions.append((sentence, most_likely, least_likely))
        #print(epoch_predictions)
    predictions_per_epoch.append(epoch_predictions)

    # # 输出当前epoch的损失信息
    # print(
    #     f"Epoch {epoch + 1}: Training Loss = {total_train_loss / len(training_data):.2f}, Validation Loss = {total_val_loss / len(validation_data):.2f}"
    # )
    # print(f"Sentence: '{sentence}'")
    # print(f"Most likely word: '[{most_likely}]';\tLeast likely word: '[{least_likely}]'\n")
    # # 输出当前epoch的损失信息
    #print(f'Epoch {epoch + 1}: Training Loss = {total_train_loss / len(training_data):.2f}, Validation Loss = {total_val_loss / len(validation_data):.2f}')
    #print( f"Epoch {epoch + 1}: Training Loss = {total_train_loss:.2f}, Validation Loss = {total_val_loss:.2f}\n")

# 记录结束时间并计算总时间
end_time = time.time()
total_time = end_time - start_time

print(f'Total training time: {total_time:.2f} seconds')


sentences = ["I ate a _____ for breakfast", "I saw a huge _____ yesterday"]
for sentence in sentences:
    most_likely, least_likely = predict_missing_word(sentence, word_to_ix, ix_to_word, model)
    print(f"Sentence: '{sentence}'")
    print(f"Most likely word: '[{most_likely}]';\tLeast likely word: '[{least_likely}]'\n")

Taskindex=['most likely','most unlikely']
# 创建一个DataFrame
Result = {
    'Epoch Number': range(1, len(train_losses) + 1),  # 添加一个从1开始的Epoch编号
    'Training Loss': train_losses,
    'Validation Loss': validation_losses,
    sentences[0]+Taskindex[0]: [pred[0][1] for pred in predictions_per_epoch],  # 从predictions中提取最可能的词
    sentences[0]+Taskindex[1]: [pred[0][2] for pred in predictions_per_epoch],  # 从predictions中提取最不可能的词
    sentences[1]+Taskindex[0]: [pred[1][1] for pred in predictions_per_epoch],  # 从predictions中提取最可能的词
    sentences[1]+Taskindex[1]: [pred[1][2] for pred in predictions_per_epoch],   # 从predictions中提取最不可能的词
'Timecost':total_time
}

df = pd.DataFrame(Result)

# 保存DataFrame到Excel文件
with pd.ExcelWriter('CBOW_training_results.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Training Results')

print("Data saved to Excel file successfully!")

# 提取模型的词嵌入
embeddings = model.embeddings.weight.to('cpu').detach().numpy()

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
low_dim_embs = tsne.fit_transform(embeddings)

# 绘制结果
plt.figure(figsize=(10, 10))
plt.title("t-SNE Visualization of Word Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# 设置图的横纵轴范围
plt.xlim(-15, 15)
plt.ylim(-15, 15)

for i, label in enumerate(word_to_ix.keys()):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.savefig('tsne_word_embeddings_CBOW.png', format='png', dpi=300, bbox_inches='tight')
#plt.show()

