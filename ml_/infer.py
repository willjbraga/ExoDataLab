import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def pre_process(df, target):
    cols_to_drop = []
    pattern = re.compile(r'\b\w*id\w*\b', re.IGNORECASE)
    for col in df.columns:
        if col == target:
            continue  
        if (df[col].dtype == 'O'):
            if pattern.search(col):
                cols_to_drop.append(col)
            else:
                df[col] = df[col].astype("category")
        elif (df[col].dtype != 'float64' and df[col].dtype != 'int64'):
            cols_to_drop.append(col)
    df.drop(columns = cols_to_drop, inplace=True)
    return df

def infer_light(df, target, dataset_name, name):
    #df, cat_features = pre_process(df, target)
    pattern = re.compile(r'\b\w*id\w*\b', re.IGNORECASE)
    cat_features = []
    for col in df.columns:
        if (df[col].dtype == 'O'):
                if not pattern.search(col):
                    cat_features.append(col)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,   
        shuffle=True    
    )

    df_train, df_val = train_test_split(
        train_df,
        test_size=0.2,   
        shuffle=True    
    )

    labels = test_df[name]

    df_train = pre_process(df_train, target)
    df_val = pre_process(df_val, target)
    test_df = pre_process(test_df, target)

    X_train_fg = df_train.drop(columns=[target])
    y_train = df_train[target]

    x_val = df_val.drop(columns=[target])
    y_val = df_val[target]

    X_query_fg = test_df.drop(columns=[target])
    y_true = test_df[target]

    with open("ml_/utils.yaml", "r") as f:
        config = yaml.safe_load(f)

    lr = str(config['model']['learning_rate'])
    depth = str(config['model']['max_depth'])
    it = str(config['model']['n_estimators'])

    model_path = os.path.join("checkpoints", dataset_name, lr, depth, it, "lightgbm_model.pkl")
    os.makedirs(os.path.join("checkpoints", dataset_name, lr, depth, it), exist_ok=True)


    if os.path.exists(model_path):
        print(f"Carregando modelo salvo em: {model_path}")
        model = joblib.load(model_path)
    else:
        print("Treinando modelo LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            learning_rate=config['model']['learning_rate'],
            subsample=config['model']['subsample'],
            colsample_bytree=config['model']['colsample_bytree'],
            random_state=config['model']['random_state'],
            boosting_type=config['model'].get('boosting_type', 'gbdt'),
            objective="multiclass" if len(np.unique(y_train)) > 2 else "binary",
            metric=config['model']['eval_metric'],
            device="cpu",
            verbosity=-1
        )
        model.fit(
            X_train_fg,
            y_train,
            eval_set=[(X_train_fg, y_train), (x_val, y_val)],
            eval_metric=config['model']['eval_metric'],
            categorical_feature=cat_features,
            callbacks=[
                lgb.log_evaluation(period=20),     # mostra a cada 20 iterações
                lgb.early_stopping(50)             # para se não melhorar em 50
            ]
        )

        joblib.dump(model, model_path)
        print(f"Modelo salvo em: {model_path}")

    # Inferência
    y_pred = model.predict(X_query_fg)
    y_proba = model.predict_proba(X_query_fg)

    # Métricas
    metrics_config = config['evaluation_metrics']['metrics']
    average = config['evaluation_metrics']['average']
    multi_class = config['evaluation_metrics']['multi-class']
    num_classes = len(np.unique(y_true))

    metrics = {}
    if "accuracy" in metrics_config:
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in metrics_config:
        metrics["Precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    if "recall" in metrics_config:
        metrics["Recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    if "f1" in metrics_config:
        metrics["F1-score"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    if "roc_auc" in metrics_config:
        if num_classes == 2:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_proba, multi_class=multi_class)

    return y_pred, metrics, y_true, labels


def main():
    infer_light()

if __name__ == "main":
    main()