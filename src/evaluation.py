from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_model(model, generator, config):
    images, labels = next(generator)
    preds = model.predict(images)
    preds_binary = (preds > 0.5).astype(int)
    
    metrics = {}
    for metric in config['evaluation']['metrics']:
        if metric == 'accuracy':
            metrics[metric] = accuracy_score(labels, preds_binary)
        elif metric == 'precision':
            metrics[metric] = precision_score(labels, preds_binary)
        elif metric == 'recall':
            metrics[metric] = recall_score(labels, preds_binary)
        elif metric == 'auc_roc':
            metrics[metric] = roc_auc_score(labels, preds)
    
    return metrics