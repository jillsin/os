import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Đọc dữ liệu từ file CSV
df = pd.read_csv("dataset/systemdata.csv")  # ← thay bằng đường dẫn file thực tế

# Kiểm tra và xử lý dữ liệu thiếu nếu có
df.dropna(inplace=True)

# Chọn các feature và nhãn
feature_cols = [
    "cpu_percent",
    "ram_percent",
    "battery_percent",
    "plugged",
    "idle_time_sec",
    "hour",
    "cpu_freq_mhz",
    "net_sent_rate",
    "net_recv_rate",
]
X = df[feature_cols]
y = df["label"]
df["plugged"] = df["plugged"].astype(int)


# Chia train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo và huấn luyện mô hình
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Ma trận nhầm lẫn
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Báo cáo phân loại
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=feature_cols,
    class_names=["High Activity", "Low Activity"],
    filled=True,
)
plt.title("Decision Tree for Predicting Low Activity Periods")
plt.show()

joblib.dump(model, "model.pkl")
# Tải lại
model = joblib.load("model.pkl")
