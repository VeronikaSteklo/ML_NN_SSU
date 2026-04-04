import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def compare_models(model_histories):
    comparison_data = []

    for model_name, history in model_histories.items():
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        test_acc = history['test_acc'] if 'test_acc' in history else 0

        avg_confidence = 0
        if 'confidence_stats' in history and history['confidence_stats']:
            avg_confidence = history['confidence_stats'].get('overall_mean', 0)

        epochs_trained = len(history['val_acc'])

        comparison_data.append({
            'Model': model_name,
            'Best Val Acc': best_val_acc,
            'Final Val Acc': final_val_acc,
            'Final Train Acc': final_train_acc,
            'Test Acc': test_acc,
            'Best Val Loss': best_val_loss,
            'Epochs Trained': epochs_trained,
            'Avg Confidence': avg_confidence
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test Acc', ascending=False)

    df_display = df_comparison.copy()
    percentage_cols = ['Best Val Acc', 'Final Val Acc', 'Final Train Acc', 'Test Acc', 'Avg Confidence']
    for col in percentage_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")

    df_display['Best Val Loss'] = df_display['Best Val Loss'].apply(lambda x: f"{x:.4f}")

    return df_comparison, df_display


def plot_confusion_matrices(models_dict, class_names, device):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]

    model_items = list(models_dict.items())

    for idx, (model_name, model_info) in enumerate(model_items):
        model = model_info['model']
        loader = model_info['loader']

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_predictions)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx])

        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.2%}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

        print(f"\n{model_name} Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))

    plt.tight_layout()
    plt.show()

    return cm
