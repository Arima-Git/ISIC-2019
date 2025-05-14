import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    balanced_accuracy_score,
    roc_curve, 
    auc,
    accuracy_score
)
import seaborn as sns
from itertools import cycle

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curves(y_true, y_score, class_names):
    """Plot ROC curves for each class."""
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert y_true to one-hot encoding
    y_true_bin = np.zeros((len(y_true), len(class_names)))
    for i, label in enumerate(y_true):
        y_true_bin[i, label] = 1
    
    # Compute ROC curve and ROC area for each class
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink'])
    
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()
    
    return roc_auc

def calculate_wacc(y_true, y_pred, class_weights):
    """Calculate Weighted Accuracy (WACC)."""
    cm = confusion_matrix(y_true, y_pred)
    wacc = 0
    for i in range(len(class_weights)):
        wacc += (cm[i, i] / cm[i, :].sum()) * class_weights[i]
    return wacc

def evaluate_model(model, test_dataset, class_names, class_weights=None):
    """Comprehensive model evaluation."""
    # Get predictions
    features, labels = next(iter(test_dataset.batch(2000)))
    labels = labels.numpy()
    predictions = model.predict(features)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions_classes)
    balanced_acc = balanced_accuracy_score(labels, predictions_classes)
    
    print('\n=== Model Evaluation Results ===')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Balanced Accuracy: {balanced_acc:.3f}')
    
    # Confusion Matrix
    print('\nConfusion Matrix:')
    cm = confusion_matrix(labels, predictions_classes)
    print(cm)
    
    # Classification Report
    print('\nClassification Report:')
    print(classification_report(labels, predictions_classes, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions_classes, class_names)
    
    # ROC curves and AUC
    roc_auc = plot_roc_curves(labels, predictions, class_names)
    print('\nAUC Scores:')
    for i, class_name in enumerate(class_names):
        print(f'{class_name}: {roc_auc[i]:.3f}')
    
    # Weighted Accuracy (WACC)
    if class_weights is not None:
        wacc = calculate_wacc(labels, predictions_classes, class_weights)
        print(f'\nWeighted Accuracy (WACC): {wacc:.3f}')
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'wacc': wacc if class_weights is not None else None
    }

if __name__ == '__main__':
    # Example usage
    # Load your model and test dataset
    # model = tf.keras.models.load_model('path_to_your_model')
    # test_dataset = ... # Your test dataset
    
    # Define class names and weights
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']
    class_weights = [0.700, 0.246, 0.953, 3.654, 1.207, 13.229, 12.517, 5.039]  # From training.py
    
    # Run evaluation
    # results = evaluate_model(model, test_dataset, class_names, class_weights) 