import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        Инициализация класса для кросс-энтропийной потери с возможностью взвешивания классов.

        :param class_weights: Вектор весов для классов (опционально)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        """
        Вычисление кросс-энтропийной потери с (или без) взвешиванием классов.

        :param y_true: Точные метки классов (вектор или одна метка)
        :param y_pred: Вероятностный вектор предсказаний
        :return: Значение потери
        """

        y_true = y_true.to(torch.long)  # Приводим метки к типу Long
        y_pred = y_pred.to(torch.float32)  # Приводим предсказания к типу Float32

        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights).float().to(y_true.device)
            loss = F.cross_entropy(y_pred, y_true, weight=class_weights)
        else:
            loss = F.cross_entropy(y_pred, y_true)
        
        return loss