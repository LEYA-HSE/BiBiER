from sklearn.metrics import recall_score, f1_score

def uar(y_true, y_pred):
    """
    Вычисление метрики UAR (Unweighted Average Recall).
    
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: UAR (Recall по всем классам без учета веса)
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0)

def war(y_true, y_pred):
    """
    Вычисление метрики WAR (Weighted Average Recall).
    
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: WAR (Recall с учетом веса классов)
    """
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def mf1(y_true, y_pred):
    """
    Вычисление метрики MF1 (Macro F1 Score).
    
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: MF1 (F1 с усреднением по всем классам)
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def wf1(y_true, y_pred):
    """
    Вычисление метрики WFI (Weighted F1 Score).
    
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: WFI (F1 с учетом веса классов)
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)