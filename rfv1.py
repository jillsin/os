import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load dữ liệu
df = pd.read_csv("dataset/systemdata.csv")

# 2. Tách features và label
X = df.drop(columns=["label"])
X = df.drop(columns=["label", "timestamp"])  # Bỏ cột không cần thiết
y = df["label"]

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Khởi tạo mô hình
rf = RandomForestClassifier(random_state=42)

# 5. Tập siêu tham số để tuning
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced"],
}

# 6. Tuning với RandomizedSearchCV
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

random_search.fit(X_train, y_train)

# 7. Đánh giá
y_pred = random_search.best_estimator_.predict(X_test)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("\n✅ Best Hyperparameters:\n", random_search.best_params_)

# 8. Lưu mô hình
joblib.dump(random_search.best_estimator_, "model_rf_tuned.pkl")
print("\n✅ Model saved to model_rf_tuned.pkl")

# 9. Vẽ biểu đồ feature importance
importances = random_search.best_estimator_.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("🎯 Feature Importance (có dùng idle_time_sec)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_with_idle.png")
plt.show()
