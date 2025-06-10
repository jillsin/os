import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dữ liệu
df = pd.read_csv("dataset/systemdata.csv")

# Tách features và label

X = df.drop(columns=["label"])
X = df.drop(columns=["label", "timestamp"])

X = df.drop(columns=["label", "idle_time_sec", "timestamp"])


y = df["label"]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Định nghĩa mô hình
rf = RandomForestClassifier(random_state=42)

# Các siêu tham số để tune
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced"],
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    scoring="f1",
    random_state=42,
    verbose=1,
)

# Huấn luyện
random_search.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = random_search.best_estimator_.predict(X_test)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# In ra siêu tham số tốt nhất
print("\n✅ Best Hyperparameters:\n", random_search.best_params_)

# Lưu mô hình
joblib.dump(random_search.best_estimator_, "model_rf_tuned.pkl")
print("\n✅ Model saved to model_rf_tuned.pkl")
importances = random_search.best_estimator_.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("🎯 Feature Importance (Không dùng idle_time_sec)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
