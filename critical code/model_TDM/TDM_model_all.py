import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
sns.set(font='Arial')

excel_file_path = 'TDMs_all.xlsx'
df = pd.read_excel(excel_file_path)
data = df

delect_col = ['formula','TDMs']
X = data.drop(delect_col,axis = 1)
y = data['TDMs'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
model = xgb.XGBRegressor(
    objective='binary:logistic',
    colsample_bytree=1,
    gamma=10,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=9,
    n_estimators=1000,
    subsample=1,
    reg_lambda=8,
    scale_pos_weight=94.7,
    eval_metric='auc',
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_class = (y_pred >= 0.1).astype(int)
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
cm = confusion_matrix(y_test, y_pred_class)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

model.save_model('model_TDM_all.json')

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-potential-DPLM", "potential-DPLM"],
            yticklabels=["Non-potential-DPLM", "potential-DPLM"])
plt.xlabel("Predicted Label", fontname='Arial')
plt.ylabel("True Label", fontname='Arial')
plt.title("Confusion matrix of the TDM model before feature engineering", fontname='Arial')
#plt.savefig("confusion_matrix_tdm_B.png", dpi=300, bbox_inches='tight')
plt.show()
#roc_data.to_csv('roc_curve_data.csv', index=False)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontname='Arial')
plt.ylabel('True Positive Rate', fontname='Arial')
plt.title('ROC Curve of model for screening potential DPLM', fontname='Arial')
plt.legend(loc='lower right', prop={'family': 'Arial'})
#plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(20, 18))
corr_matrix = X.corr()
sns.heatmap(
    corr_matrix,
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    cmap='coolwarm',
    center=0,
    annot=False,
    square=True
)
plt.title('Feature Correlation Matrix of TDM model', fontname='Arial', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('feature_correlation_heatmap.png', dpi=300)
plt.show()