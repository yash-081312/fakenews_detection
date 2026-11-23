"""
train.py
Model training, evaluation, comparison, figure generation, and serialization.
Implements the 5 benchmark algorithms for Fake News Detection.
"""

import os
import sys

# Ensure root project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from src.data_loader import prepare_and_cache_dataset, get_train_test_split, LABEL_MAP, INV_LABEL_MAP

# Set plotting styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "results/figures"):
    """
    Generates and saves exploratory data analysis (EDA) plots:
    1. Class Distribution
    2. Text Length / Word Count Distribution
    3. Word Clouds for Real and Fake news
    """
    os.makedirs(output_dir, exist_ok=True)
    print("[EDA] Generating Exploratory Data Analysis figures...")

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(7, 5))
    counts = df['label_name'].value_counts()
    colors = ['#e74c3c' if label == 'Fake' else '#2ecc71' for label in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)
    plt.title("Class Balance Distribution (Fake vs Real)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("News Category", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    for p, count in zip(ax.patches, counts.values):
        pct = (count / len(df)) * 100
        ax.annotate(f"{count:,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    class_dist_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {class_dist_path}")

    # 2. Text Length Distribution Histogram / KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='cleaned_word_count', hue='label_name', kde=True,
                 palette={'Fake': '#e74c3c', 'Real': '#2ecc71'}, bins=60, alpha=0.5)
    plt.title("Distribution of Article Word Counts (Cleaned)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Word Count per Article", fontsize=12)
    plt.ylabel("Article Frequency", fontsize=12)
    plt.xlim(0, 1000)
    word_dist_path = os.path.join(output_dir, "text_length_distribution.png")
    plt.savefig(word_dist_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {word_dist_path}")

    # 3. WordCloud for Real News
    print("[EDA] Generating WordCloud for Real news...")
    real_text = " ".join(df[df['label_name'] == 'Real']['cleaned_text'])
    wc_real = WordCloud(width=800, height=450, background_color='white',
                        colormap='Greens', max_words=150, random_state=42).generate(real_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc_real, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Frequent Terms - REAL News", fontsize=16, fontweight='bold', pad=14)
    wc_real_path = os.path.join(output_dir, "wordcloud_real.png")
    plt.savefig(wc_real_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {wc_real_path}")

    # 4. WordCloud for Fake News
    print("[EDA] Generating WordCloud for Fake news...")
    fake_text = " ".join(df[df['label_name'] == 'Fake']['cleaned_text'])
    wc_fake = WordCloud(width=800, height=450, background_color='white',
                        colormap='Reds', max_words=150, random_state=42).generate(fake_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Frequent Terms - FAKE News", fontsize=16, fontweight='bold', pad=14)
    wc_fake_path = os.path.join(output_dir, "wordcloud_fake.png")
    plt.savefig(wc_fake_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {wc_fake_path}")


def get_model_definitions():
    """
    Returns the 5 benchmark algorithms evaluated in the project.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver='lbfgs'
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.1
        ),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000),
            method='sigmoid'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=35,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    }
    return models


def train_and_evaluate_all():
    """
    End-to-end training and evaluation pipeline.
    """
    models_dir = "models"
    figures_dir = "results/figures"
    results_dir = "results"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    df = prepare_and_cache_dataset()

    # 2. Generate EDA plots
    generate_eda_figures(df, figures_dir)

    # 3. Train/Test Split
    X_train_raw, X_test_raw, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)

    # 4. Feature Extraction: TF-IDF Vectorizer
    print("\n[Feature Extraction] Fitting TF-IDF Vectorizer (max_features=10,000, ngrams=1-2)...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    print(f"[Feature Extraction] TF-IDF Train matrix shape: {X_train_tfidf.shape}")

    # Save TF-IDF Vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf, tfidf_path)
    print(f"[Feature Extraction] Saved Vectorizer: {tfidf_path}")

    # 5. Model Training & Evaluation
    models = get_model_definitions()
    results = []
    confusion_matrices = {}
    roc_curves_data = {}
    reports_text = []

    print("\n" + "="*70)
    print("BENCHMARK MODEL TRAINING & EVALUATION")
    print("="*70)

    for name, model in models.items():
        print(f"\n[Training] Model: {name}...")
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_duration = time.time() - start_time

        # Inference
        y_pred = model.predict(X_test_tfidf)
        
        # Probabilities for ROC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test_tfidf)
        else:
            y_proba = y_pred

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr, tpr, roc_auc)

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "TrainTimeSec": round(train_duration, 2)
        })

        # Save model checkpoint
        slug = name.lower().replace(" ", "_")
        model_save_path = os.path.join(models_dir, f"{slug}.pkl")
        joblib.dump(model, model_save_path)

        rep = f"=== Classification Report: {name} ===\n"
        rep += classification_report(y_test, y_pred, target_names=["Real", "Fake"], digits=4)
        rep += f"ROC-AUC Score: {roc_auc:.4f} | Training Time: {train_duration:.2f}s\n\n"
        reports_text.append(rep)

        print(f" -> Accuracy:  {acc*100:.2f}% | F1: {f1*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}%")

    # 6. Save Reports & Results Table
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[Results] Saved metrics summary table to: {results_csv_path}")

    reports_path = os.path.join(results_dir, "classification_reports.txt")
    with open(reports_path, "w", encoding="utf-8") as f:
        f.writelines(reports_text)
    print(f"[Results] Saved full classification reports to: {reports_path}")

    # Identify Best Model based on F1-Score
    best_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_model_name = best_row["Algorithm"]
    best_slug = best_model_name.lower().replace(" ", "_")
    best_model_obj = joblib.load(os.path.join(models_dir, f"{best_slug}.pkl"))
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))
    print(f"\n[Model Selection] Best Model: {best_model_name} (F1-Score: {best_row['F1-Score']*100:.2f}%)")
    print(f"[Model Selection] Saved as models/best_model.pkl")

    # 7. Generate Comparison Plots
    _plot_model_comparison(results_df, figures_dir)
    _plot_confusion_matrices(confusion_matrices, figures_dir)
    _plot_roc_curves(roc_curves_data, figures_dir)

    print("\n" + "="*70)
    print("FINAL SUMMARY METRICS (FOR SLIDE 9 & REPORT):")
    print("="*70)
    summary_display = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        summary_display[col] = (summary_display[col] * 100).map(lambda x: f"{x:.2f}%")
    print(summary_display.to_string(index=False))

    return results_df


def _plot_model_comparison(results_df: pd.DataFrame, output_dir: str):
    """
    Plots multi-metric grouped bar chart comparing algorithms.
    """
    df_melted = results_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted,
        x="Algorithm",
        y="Score",
        hue="Metric",
        palette="viridis"
    )
    plt.title("Comparative Performance Across 5 ML Algorithms", fontsize=15, fontweight="bold", pad=12)
    plt.ylim(0.85, 1.02)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Classification Algorithm", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        h = p.get_height()
        if h > 0.01:
            ax.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.005),
                        ha='center', va='bottom', fontsize=8.5, rotation=45)

    comp_path = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {comp_path}")


def _plot_confusion_matrices(cms: dict, output_dir: str):
    """
    Plots a 1x5 or 2x3 grid of confusion matrices.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"]
        )
        ax.set_title(f"{name}\nAcc: {np.trace(cm)/np.sum(cm)*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Normalized Confusion Matrices on Test Set (1,973 samples)", fontsize=15, fontweight="bold", y=1.05)
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {cm_path}")


def _plot_roc_curves(roc_data: dict, output_dir: str):
    """
    Plots Receiver Operating Characteristic (ROC) curves for all models.
    """
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=15, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {roc_path}")


if __name__ == "__main__":
    train_and_evaluate_all()
