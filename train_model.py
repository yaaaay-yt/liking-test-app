import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- 修正点：生成更真实的模拟数据 ---
np.random.seed(42)
n_samples = 1000

# 1. 为每个样本生成一个潜在的“喜欢程度”分数 (0.0 到 1.0)
#    这个分数将决定该样本在所有问题上的得分高低
true_liking_level = np.random.rand(n_samples)  # 0.0 (完全不喜欢) to 1.0 (非常喜欢)

# 2. 基于“喜欢程度”生成20道题的得分
#    喜欢程度越高，得分越高的概率越大
X_data = np.zeros((n_samples, 20))
for i in range(n_samples):
    # 使用sigmoid函数将喜欢程度映射到一个更集中的区间，然后乘以4再加1，得到1-5分
    # 这样可以保证喜欢程度高的样本，得分普遍偏高
    base_score = 1 + 4 * (1 / (1 + np.exp(-10 * (true_liking_level[i] - 0.5))))
    # 为每个题目分数添加一些随机噪声，使其更真实
    X_data[i] = np.random.normal(loc=base_score, scale=0.3, size=20)
    # 确保分数在1-5之间
    X_data[i] = np.clip(X_data[i], 1, 5)
    # 四舍五入到整数分
    X_data[i] = np.round(X_data[i])

X_data = X_data.astype(int)
total_scores = X_data.sum(axis=1)

# 3. 按总分划分情感等级（与之前逻辑一致）
y_data = pd.cut(
    total_scores,
    bins=[0, 40, 60, 75, 90, 100],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# 转换为DataFrame
data = pd.DataFrame(X_data, columns=[f'q{i+1}' for i in range(20)])
data['label'] = y_data
data['total_score'] = total_scores # 方便查看

# --- 数据生成结束 ---

# 2. 划分训练集和测试集
X = data.drop(['label', 'total_score'], axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型测试集准确率：{accuracy:.2f}")

# 打印一些预测示例来验证
print("\n--- 预测示例 ---")
sample_indices = [0, 100, 200, 300, 400]
for idx in sample_indices:
    sample_score = total_scores[idx]
    true_label = y_data[idx]
    pred_label = model.predict(X.iloc[[idx]])[0]
    print(f"样本 {idx}: 总分={sample_score}, 真实标签={true_label}, 预测标签={pred_label}")


# 5. 保存模型和标签映射
joblib.dump(model, 'emotion_model.pkl')
label_mapping = {0: "普通朋友", 1: "好感阶段", 2: "有点喜欢", 3: "比较喜欢", 4: "非常喜欢"}
joblib.dump(label_mapping, 'label_mapping.pkl')
print("\n模型和标签映射已保存！")