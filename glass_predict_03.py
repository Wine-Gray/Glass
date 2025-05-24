import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. 读取数据集
df = pd.read_csv("ceramic_types_100.csv")  # 确保路径正确

# 2. 选择输入特征和目标变量
features = ['Al2O3(%)', 'SiO2(%)', 'Fe2O3(%)', 'MgO(%)']
target = '热导率(W/m·K)'

X = df[features]
y = df[target]

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 建立并训练随机森林回归模型
rf_thermal = RandomForestRegressor(n_estimators=100, random_state=42)
rf_thermal.fit(X_train, y_train)

# 5. 测试集评估
y_pred = rf_thermal.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("热导率预测——测试集结果：")
print(f"  MAE: {mae:.4f}")
print(f"  R2 : {r2:.4f}")

# 6. 保存模型
joblib.dump(rf_thermal, 'rf_thermal_model.pkl')
print("模型已保存为 rf_thermal_model.pkl")

# 7. （可选）新配方预测
# 例：Al2O3=60, SiO2=30, Fe2O3=1.0, MgO=1.0
new_input = [[60, 30, 1.0, 1.0]]
pred_thermal = rf_thermal.predict(new_input)[0]
print(f"\n新配方预测热导率: {pred_thermal:.3f} W/m·K")
