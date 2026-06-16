import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

base_data_path = r'D:\学习工作\陈'
result_save_path = os.path.join(base_data_path, '结果输出')

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
    print(f"已创建结果保存文件夹：{result_save_path}")

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gbk')

print("正在读取数据...")
datasets = {
    'Train': read_csv_safe(os.path.join(base_data_path, '训练集.csv')),
    'Test': read_csv_safe(os.path.join(base_data_path, '测试集.csv'))
}

all_features = [set(df.columns) for df in datasets.values()]
common_features = list(set.intersection(*all_features) - {'status'})

X_dict, y_dict = {}, {}
ss = StandardScaler()

X_train = datasets['Train'][common_features]
y_train = datasets['Train']['status']
X_dict['Train'] = ss.fit_transform(X_train)
y_dict['Train'] = y_train

X_test = datasets['Test'][common_features]
y_test = datasets['Test']['status']
X_dict['Test'] = ss.transform(X_test)
y_dict['Test'] = y_test

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
imb_ratio = float(neg_count) / pos_count
print(f"\n[数据读取完毕] 训练集真实分布 -> 阴性: {neg_count}, 阳性: {pos_count}")
print(f"[参数设定] scale_pos_weight: {imb_ratio:.2f}\n")

best_params = {
    'XGBoost': {
        'subsample': 0.8, 'reg_lambda': 0.2, 'n_estimators': 800, 'min_child_weight': 10, 
        'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.2, 'colsample_bytree': 0.6, 
        'scale_pos_weight': imb_ratio, 'objective': 'binary:logistic', 'eval_metric': 'auc',
        'random_state': 42, 'nthread': -1
    },
    'HistGBDT': {
        'max_iter': 300, 'learning_rate': 0.05, 'max_depth': 5, 'l2_regularization': 1.0, 
        'class_weight': 'balanced', 'random_state': 42
    },
    'LightGBM': {
        'subsample_freq': 1, 'subsample': 0.8, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 
        'num_leaves': 31, 'min_child_samples': 20, 'max_depth': 5, 'learning_rate': 0.02, 
        'colsample_bytree': 0.8, 'random_state': 2025, 'class_weight': 'balanced', 
        'n_jobs': -1, 'verbose': -1
    },
    'Random Forest': {
        'n_estimators': 300, 'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 5, 
        'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1, 
        'class_weight': 'balanced'
    },
    'AdaBoost': {
        'n_estimators': 150, 'learning_rate': 0.05, 'algorithm': 'SAMME', 'random_state': 42
    },
    'AdaBoost_base_estimator': {
        'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'gini', 
        'random_state': 42, 'class_weight': 'balanced'
    },
    'Logistic Regression': {
        'random_state': 2024, 'tol': 1e-5, 'penalty': 'l2', 'C': 0.8, 'solver': 'liblinear', 
        'class_weight': 'balanced'
    },
    'Decision Tree': {
        'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 3, 
        'random_state': 42, 'class_weight': 'balanced'
    },
    'Naive Bayes': {
        'var_smoothing': 1e-9
    }
}

models = {
    'XGBoost': xgb.XGBClassifier(**best_params['XGBoost']),
    'GBDT (Hist)': HistGradientBoostingClassifier(**best_params['HistGBDT']),
    'LightGBM': lgb.LGBMClassifier(**best_params['LightGBM']),
    'Random Forest': RandomForestClassifier(**best_params['Random Forest']),
    'AdaBoost': AdaBoostClassifier(**best_params['AdaBoost'], estimator=DecisionTreeClassifier(**best_params['AdaBoost_base_estimator'])),
    'Logistic Regression': LogisticRegression(**best_params['Logistic Regression']),
    'Decision Tree': DecisionTreeClassifier(**best_params['Decision Tree']),
    'Naive Bayes': GaussianNB(**best_params['Naive Bayes'])
}

print(" 开始训练模型...")
trained_models = {}
for model_name, model in models.items():
    print(f"   -> 训练: {model_name}")
    model.fit(X_dict['Train'], y_dict['Train'])
    trained_models[model_name] = model
print(" 所有模型训练完成！\n")

def calculate_simplified_metrics(X, y, model, model_name, dataset_name, n_bootstraps=0):
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1].clip(1e-7, 1 - 1e-7)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0] if y.nunique() == 1 else 0)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    
    accuracy = accuracy_score(y, y_pred)
    youden_index = sensitivity + specificity - 1
    _, _, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y, y_probs) if len(np.unique(y)) > 1 else 0.5

    auc_ci = (np.nan, np.nan)
    if n_bootstraps > 0 and len(y) >= 2 and len(np.unique(y)) > 1:
        rng = np.random.default_rng(42)
        y_np = np.array(y)
        y_probs_np = np.array(y_probs)
        indices = rng.choice(len(y_np), size=(n_bootstraps, len(y_np)), replace=True)
        
        auc_boot = []
        for idx in indices:
            y_boot = y_np[idx]
            if len(np.unique(y_boot)) > 1:
                auc_boot.append(roc_auc_score(y_boot, y_probs_np[idx]))
                
        if len(auc_boot) > 0:
            auc_ci = np.quantile(auc_boot, [0.025, 0.975])

    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'AUC': auc,
        'AUC_95CI': f'({auc_ci[0]:.4f}, {auc_ci[1]:.4f})' if not np.isnan(auc_ci[0]) else '(NaN, NaN)',
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Youden_Index': youden_index,
        'F1_Score': f1,
        'Accuracy': accuracy
    }

all_results = []
print(" 开始计算模型评估指标...")
for model_name, model in trained_models.items():
    print(f"   -> 评估并计算CI: {model_name}")
    for dataset_name in ['Train', 'Test']:
        metrics = calculate_simplified_metrics(
            X=X_dict[dataset_name],
            y=y_dict[dataset_name],
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            n_bootstraps=1000 
        )
        all_results.append(metrics)

result_cols = [
    'Model', 'Dataset', 'AUC', 'AUC_95CI', 'Sensitivity', 
    'Specificity', 'Youden_Index', 'F1_Score', 'Accuracy'
]
result_df = pd.DataFrame(all_results)[result_cols]

csv_save_path = os.path.join(result_save_path, '8模型_成稿版_精简指标汇总.csv')
result_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

print(f"\n 任务完成！指标文件已保存至：\n{csv_save_path}")

print("\n===  【测试集】表现排行榜 ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}' if not pd.isna(x) else 'NaN')
test_results = result_df[result_df['Dataset'] == 'Test'].sort_values(by='AUC', ascending=False)
print(test_results.to_string(index=False))



















import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

model_styles = {
    'XGBoost': {'color': '#e41a1c', 'linestyle': '-'},
    'GBDT (Hist)': {'color': '#377eb8', 'linestyle': '-'},
    'LightGBM': {'color': '#4daf4a', 'linestyle': '-'},
    'Random Forest': {'color': '#984ea3', 'linestyle': '-'},
    'AdaBoost': {'color': '#ff7f00', 'linestyle': '--'},
    'Logistic Regression': {'color': '#a65628', 'linestyle': '--'},
    'Decision Tree': {'color': '#f781bf', 'linestyle': '-.'},
    'Naive Bayes': {'color': '#999999', 'linestyle': '-.'}
}

def plot_and_save_roc(dataset_name, X, y, models_dict, save_dir):
    plt.figure(figsize=(8, 8), dpi=300)
    
    for model_name, model in models_dict.items():
        y_probs = model.predict_proba(X)[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_probs)
        roc_auc = auc(fpr, tpr)
        
        style = model_styles.get(model_name, {'color': 'black', 'linestyle': '-'})
        
        plt.plot(fpr, tpr, color=style['color'], linestyle=style['linestyle'], lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title(f'ROC Curves on {dataset_name} Set', fontweight='bold', pad=15)
    
    plt.legend(loc="lower right", frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    pdf_path = os.path.join(save_dir, f'ROC_Curve_{dataset_name}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f" {dataset_name} 集的 ROC 曲线已保存至：{pdf_path}")

print(" 正在绘制高分辨率 ROC 曲线...")

plot_and_save_roc(
    dataset_name='Training', 
    X=X_dict['Train'], 
    y=y_dict['Train'], 
    models_dict=trained_models, 
    save_dir=result_save_path
)

plot_and_save_roc(
    dataset_name='Testing', 
    X=X_dict['Test'], 
    y=y_dict['Test'], 
    models_dict=trained_models, 
    save_dir=result_save_path
)

print("\n 全部绘图任务完成！")














import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

model_styles = {
    'XGBoost': {'color': '#e41a1c', 'linestyle': '-'},
    'GBDT (Hist)': {'color': '#377eb8', 'linestyle': '-'},
    'LightGBM': {'color': '#4daf4a', 'linestyle': '-'},
    'Random Forest': {'color': '#984ea3', 'linestyle': '-'},
    'AdaBoost': {'color': '#ff7f00', 'linestyle': '--'},
    'Logistic Regression': {'color': '#a65628', 'linestyle': '--'},
    'Decision Tree': {'color': '#f781bf', 'linestyle': '-.'},
    'Naive Bayes': {'color': '#999999', 'linestyle': '-.'}
}

def plot_clinical_curves(dataset_name, X_eval, y_eval, X_train, y_train, models_dict, save_dir):
    fig_cal, ax_cal = plt.subplots(figsize=(8, 8), dpi=300)
    fig_dca, ax_dca = plt.subplots(figsize=(8, 8), dpi=300)
    
    ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", lw=2)
    
    pt_arr = np.linspace(0.01, 0.60, 60) 
    y_eval_np = np.array(y_eval)
    N = len(y_eval_np)
    
    TP_all = np.sum(y_eval_np == 1)
    FP_all = np.sum(y_eval_np == 0)
    prevalence = TP_all / N
    nb_all = prevalence - ((FP_all / N) * (pt_arr / (1 - pt_arr)))
    
    ax_dca.plot(pt_arr, np.zeros_like(pt_arr), color='black', lw=2, label='Treat None (Net Benefit = 0)')
    ax_dca.plot(pt_arr, nb_all, color='gray', lw=2, linestyle=':', label='Treat All')

    print(f"[{dataset_name}] 正在进行概率重校准并绘图...")
    for model_name, model in models_dict.items():
        raw_train_probs = model.predict_proba(X_train)[:, 1]
        raw_eval_probs = model.predict_proba(X_eval)[:, 1]
        
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_train_probs, y_train)
        calibrated_probs = iso.transform(raw_eval_probs)
        
        style = model_styles.get(model_name, {'color': 'black', 'linestyle': '-'})
        
        prob_true, prob_pred = calibration_curve(y_eval, calibrated_probs, n_bins=10, strategy='quantile')
        ax_cal.plot(prob_pred, prob_true, "s-", color=style['color'], linestyle=style['linestyle'],
                    label=model_name, lw=2, markersize=5)
                    
        nb_model = []
        for pt in pt_arr:
            preds = (calibrated_probs >= pt).astype(int)
            TP = np.sum((preds == 1) & (y_eval_np == 1))
            FP = np.sum((preds == 1) & (y_eval_np == 0))
            odds = pt / (1 - pt)
            nb = (TP / N) - (FP / N) * odds
            nb_model.append(nb)
            
        ax_dca.plot(pt_arr, nb_model, color=style['color'], linestyle=style['linestyle'],
                    label=model_name, lw=2)

    ax_cal.set_xlim([-0.05, 1.05])
    ax_cal.set_ylim([-0.05, 1.05])
    ax_cal.set_xlabel('Mean Predicted Probability (Recalibrated)', fontweight='bold')
    ax_cal.set_ylabel('Fraction of Positives (Actual)', fontweight='bold')
    ax_cal.set_title(f'Calibration Curves on {dataset_name} Set', fontweight='bold', pad=15)
    ax_cal.legend(loc="lower right", frameon=True, edgecolor='black')
    ax_cal.grid(True, linestyle=':', alpha=0.6)
    fig_cal.tight_layout()
    pdf_cal = os.path.join(save_dir, f'Calibration_Curve_{dataset_name}_Recalibrated.pdf')
    fig_cal.savefig(pdf_cal, format='pdf', bbox_inches='tight')
    plt.close(fig_cal)
    
    ax_dca.set_xlim([0.0, 0.6]) 
    ax_dca.set_ylim([-0.02, prevalence + 0.03]) 
    ax_dca.set_xlabel('Threshold Probability', fontweight='bold')
    ax_dca.set_ylabel('Net Benefit', fontweight='bold')
    ax_dca.set_title(f'Decision Curve Analysis (DCA) on {dataset_name} Set', fontweight='bold', pad=15)
    ax_dca.legend(loc="upper right", frameon=True, edgecolor='black')
    ax_dca.grid(True, linestyle=':', alpha=0.6)
    fig_dca.tight_layout()
    pdf_dca = os.path.join(save_dir, f'DCA_Curve_{dataset_name}_Recalibrated.pdf')
    fig_dca.savefig(pdf_dca, format='pdf', bbox_inches='tight')
    plt.close(fig_dca)

print("\n 开始绘制 Calibration 与 DCA 曲线...")

plot_clinical_curves('Training', X_dict['Train'], y_dict['Train'], 
                     X_dict['Train'], y_dict['Train'], trained_models, result_save_path)

plot_clinical_curves('Testing', X_dict['Test'], y_dict['Test'], 
                     X_dict['Train'], y_dict['Train'], trained_models, result_save_path)

print("\n 全部绘图任务完成！")

























import shap
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = result_save_path 

target_model_name = 'XGBoost'  
model_to_explain = trained_models[target_model_name]

X_test_raw = datasets['Test'][common_features]

print(f" 开始为 {target_model_name} 模型计算 SHAP 值...")

explainer = shap.TreeExplainer(model_to_explain)
shap_values = explainer.shap_values(X_dict['Test']) 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig_summary = plt.figure(figsize=(10, 8), dpi=300)

shap.summary_plot(
    shap_values, 
    features=X_test_raw, 
    feature_names=common_features, 
    max_display=16, 
    plot_type="dot", 
    show=False
)

ax = plt.gca()
ax.set_title(f'SHAP Summary Plot ({target_model_name})', fontweight='bold', fontsize=16, pad=20)
ax.set_xlabel('SHAP value (Impact on model output)', fontweight='bold', fontsize=14)
plt.tight_layout()

pdf_path_summary = os.path.join(save_dir, f'SHAP_Summary_{target_model_name}.pdf')
plt.savefig(pdf_path_summary, format='pdf', bbox_inches='tight')
plt.close(fig_summary)
print(f" SHAP 蜂群图 (Summary Plot) 已保存至：{pdf_path_summary}")

fig_bar = plt.figure(figsize=(10, 8), dpi=300)

shap.summary_plot(
    shap_values, 
    features=X_test_raw, 
    feature_names=common_features, 
    max_display=16, 
    plot_type="bar", 
    color="#377eb8", 
    show=False
)

ax_bar = plt.gca()
ax_bar.set_title(f'SHAP Feature Importance ({target_model_name})', fontweight='bold', fontsize=16, pad=20)
ax_bar.set_xlabel('mean(|SHAP value|) (Average impact on model output magnitude)', fontweight='bold', fontsize=14)
plt.tight_layout()

pdf_path_bar = os.path.join(save_dir, f'SHAP_Bar_{target_model_name}.pdf')
plt.savefig(pdf_path_bar, format='pdf', bbox_inches='tight')
plt.close(fig_bar)
print(f" SHAP 条形图 (Bar Plot) 已保存至：{pdf_path_bar}")

print("\n SHAP 绘图任务全部完成！")



















print("\n 正在测试集中寻找最具代表性的高危患者...")

y_test_probs = model_to_explain.predict_proba(X_dict['Test'])[:, 1]

X_test_raw = X_test_raw.reset_index(drop=True) 
patient_idx = np.argmax(y_test_probs) 
patient_risk = y_test_probs[patient_idx]

print(f" 已锁定靶标患者 (Index: {patient_idx})，预测患病概率: {patient_risk:.2%}")

shap_expl_object = explainer(X_dict['Test'])
shap_expl_object.data = X_test_raw.values 
shap_expl_object.feature_names = common_features

fig_waterfall = plt.figure(figsize=(10, 8), dpi=300)

shap.plots.waterfall(
    shap_expl_object[patient_idx], 
    max_display=16, 
    show=False
)

ax_water = plt.gca()
ax_water.set_title('SHAP Waterfall Plot (High-Risk Patient Profile)', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()

pdf_path_waterfall = os.path.join(save_dir, f'SHAP_Waterfall_{target_model_name}_Patient{patient_idx}.pdf')
plt.savefig(pdf_path_waterfall, format='pdf', bbox_inches='tight')
plt.close(fig_waterfall)
print(f" 单样本瀑布图 (Waterfall Plot) 已保存至：{pdf_path_waterfall}")


base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

plt.figure(figsize=(14, 4), dpi=300)

shap.force_plot(
    base_value, 
    shap_values[patient_idx, :], 
    X_test_raw.iloc[patient_idx, :], 
    matplotlib=True, 
    show=False,
    text_rotation=30, 
    contribution_threshold=0 
)

fig_force = plt.gcf()
plt.title('SHAP Force Plot (Individual Risk Contributors)', fontweight='bold', fontsize=16, y=1.5)

pdf_path_force = os.path.join(save_dir, f'SHAP_Force_{target_model_name}_Patient{patient_idx}.pdf')
fig_force.savefig(pdf_path_force, format='pdf', bbox_inches='tight')
plt.close(fig_force)
print(f"单样本力图 (Force Plot) 已保存至：{pdf_path_force}")

print("\n 所有个案解释图表绘制完成！")

