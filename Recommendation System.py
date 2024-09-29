# recommendation_system.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 示例數據
data = {
    'user_id': np.random.randint(1, 100, 100),
    'product_id': np.random.randint(1, 50, 100),
    'rating': np.random.randint(1, 6, 100)
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 分割數據集
X = df[['user_id', 'product_id']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))