import torch
import torch.nn as nn
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载GloVe词向量
def load_glove_embeddings(path, word_to_ix, embedding_dim):
    embeddings = np.random.rand(len(word_to_ix), embedding_dim)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_ix:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word_to_ix[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float)

# 加载并预处理文本
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
    #sentences = re.split(r"['.!?]", text)
    sentences = re.split(r'[.!?;"]+', text)
    processed_sentences = []
    for sentence in sentences:
        sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)
        words = sentence.split()
        if words:
            processed_sentences.extend(["<start>"] + words + ["<end>"])
    return processed_sentences

# 生成Skip-gram训练数据
def generate_skipgram_data(raw_text, context_size, word_to_ix):
    data = []
    for i in range(context_size, len(raw_text) - context_size):
        target = word_to_ix[raw_text[i]]
        contexts = [word_to_ix[raw_text[j]] for j in range(i - context_size, i + context_size + 1) if j != i]
        for context in contexts:
            data.append((torch.tensor([target], dtype=torch.long), torch.tensor([context], dtype=torch.long)))
    return data

# 定义Skip-gram模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_word_indices):
        embeds = self.embeddings(input_word_indices)
        out = self.linear(embeds)
        log_probs = nn.functional.log_softmax(out, dim=-1)
        return log_probs

# 参数设置
CONTEXT_SIZE = 2
EMBEDDING_DIM = 100
GLOVE_PATH = 'glove.6B.100d.txt'
TRAIN_TEXT_PATH = 'text_train.txt'
TEST_TEXT_PATH = 'text_test.txt'

# 加载和处理文本数据
train_text = load_text(TRAIN_TEXT_PATH)
test_text = load_text(TEST_TEXT_PATH)
vocab = set(train_text + test_text)
vocab_size = len(vocab)
# word_to_ix = {word: ix for ix, word in enumerate(vocab)}
# ix_to_word = {ix: word for word in word_to_ix}

word_to_ix = {word: ix for ix, word in enumerate(vocab)}
ix_to_word = {ix: word for ix, word in enumerate(vocab)}

# 加载词向量
pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word_to_ix, EMBEDDING_DIM)

# 创建模型
model = SkipGram(vocab_size, EMBEDDING_DIM, pretrained_embeddings)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 准备数据
train_data = generate_skipgram_data(train_text, CONTEXT_SIZE, word_to_ix)
test_data = generate_skipgram_data(test_text, CONTEXT_SIZE, word_to_ix)

# 训练模型
train_losses = []
validation_losses = []
predictions_per_epoch = []
import time

# 在训练之前记录开始时间
start_time = time.time()

for epoch in range(100):
    model.train()
    total_train_loss = 0
    for target, context in train_data:
        log_probs = model(target)
        loss = loss_function(log_probs, context)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_losses.append(total_train_loss / len(train_data))

    # 计算验证损失
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for target, context in test_data:
            log_probs = model(target)
            loss = loss_function(log_probs, context)
            total_val_loss += loss.item()
    validation_losses.append(total_val_loss / len(test_data))

    predictions_per_epoch_temp = {}
    context_words = ["croissant", "cloud"]
    for context_word in context_words:
        context_word_idx = torch.tensor([word_to_ix[context_word]], dtype=torch.long)
        log_probs = model(context_word_idx)
        top_context_indices = log_probs.topk(4).indices.tolist()
        predicted_context_words = [ix_to_word[idx] for idx in top_context_indices[0]]
        predictions_per_epoch_temp[context_word] = predicted_context_words
    predictions_per_epoch.append(predictions_per_epoch_temp)

    # print(f'Epoch {epoch+1}, Training Loss: {train_losses[-1]}, Validation Loss: {validation_losses[-1]}')
    # print(f'Context predictions for "{context_word}": {predicted_context_words}')

# 记录结束时间并计算总时间
end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')

# 构造DataFrame时，处理每个词的预测结果
results = {
    'Epoch Number': range(1, len(train_losses) + 1),
    'Training Loss': train_losses,
    'Validation Loss': validation_losses,
    'Timecost':total_time
}

for word in ['croissant', 'cloud']:
    results[f'{word} Predictions'] = [epoch[word] for epoch in predictions_per_epoch]

results_df = pd.DataFrame(results)
with pd.ExcelWriter('Skipgram_training_results.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Training Results')


# t-SNE可视化
embeddings = model.embeddings.weight.to('cpu').detach().numpy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
embeds_tsne = tsne.fit_transform(embeddings)
plt.figure(figsize=(10, 10))
plt.title("t-SNE Visualization of Word Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
# 设置图的横纵轴范围
plt.xlim(-15, 15)
plt.ylim(-15, 15)


for i, label in enumerate(word_to_ix.keys()):
    x, y = embeds_tsne[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.savefig('tsne_word_embeddings_Skipgram.png', dpi=300)
plt.show()
