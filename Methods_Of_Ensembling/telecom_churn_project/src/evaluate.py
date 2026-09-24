import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

def make_confusion_matrix(y_true, y_pred, model_name, by, ax=None):
    allowed_by = ['all', 'true', 'pred', None]
    if by not in allowed_by:
        raise ValueError(
            f"Invalid value for 'by': '{by}'. Allowed options are: 'all', 'true', 'pred', or None."
        )
    cm = confusion_matrix(y_true = y_true, y_pred=y_pred, normalize=by)
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    sns.heatmap(
        data=cm,
        cmap='Blues',
        linecolor='k',
        linewidths=1,
        annot=True,
        annot_kws={'size': 14},
        fmt='.2f',
        ax=ax,  
    )

    ax.set_title(f'{model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    
    if created_fig:
        plt.tight_layout()
        plt.show()



def make_roc(y_true, y_proba, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=1)

    plt.plot(fpr, tpr, lw=2, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.show()

def make_pr(y_true, y_proba, model_name):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, y_score=y_proba, pos_label=1)
    plt.plot(recall, precision, lw=2, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True)
    # plt.show()

def plot_shap(pipeline, customer):
    preprocessor = pipeline.named_steps['preprocessing']
    rf_model = pipeline.named_steps['model']

    preprocessor.verbose_feature_names_out = False

    transformed_data = preprocessor.transform(customer)
    if hasattr(transformed_data, 'toarray'):
        transformed_data = transformed_data.toarray()

    features = preprocessor.get_feature_names_out()
    customer_df = pd.DataFrame(data=transformed_data, columns=features)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(customer_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.title('Customer Churn Risk Factors', fontsize=12, pad=15)
    plt.tight_layout()

    return fig    
