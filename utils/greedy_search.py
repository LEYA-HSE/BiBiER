# coding: utf-8

import copy
import logging
from typing import Any


def format_result_box(step_num, param_name, candidate, fixed_params, dev_metrics, is_best=False):
    title = f"Шаг {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]
    metric_lines = []
    for k in ["uar", "war", "mf1", "wf1", "loss", "mean"]:
        if k in dev_metrics:
            val = dev_metrics[k]
            if isinstance(val, float):
                metric_lines.append(f"{k.upper():4} = {val:.4f}")
            else:
                metric_lines.append(f"{k.upper():4} = {val}")

    if is_best and metric_lines and "MEAN" in metric_lines[-1].upper():
        metric_lines[-1] += " ✅"

    content_lines = []
    content_lines.append(title)
    content_lines.append("  Фиксировано:")
    for line in fixed_lines:
        content_lines.append(f"    {line}")
    content_lines.append("  Результаты:")
    for line in metric_lines:
        content_lines.append(f"    {line}")

    max_width = max(len(line) for line in content_lines)
    border_top = "┌" + "─" * (max_width + 2) + "┐"
    border_bot = "└" + "─" * (max_width + 2) + "┘"

    box = [border_top]
    for line in content_lines:
        box.append(f"│ {line.ljust(max_width)} │")
    box.append(border_bot)

    return "\n".join(box)


def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any]
):
    current_best_params = copy.deepcopy(default_values)
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Жадный (поэтапный) перебор гиперпараметров (Dev-based) ===\n")
        f.write(f"Модель: {model_name}\n")

    for i, param_name in enumerate(all_param_names):
        candidates = param_grid[param_name]
        tried_value = current_best_params[param_name]

        if i == 0:
            candidates_to_try = candidates
        else:
            candidates_to_try = [v for v in candidates if v != tried_value]

        best_val_for_param = tried_value
        best_metric_for_param = float("-inf")

        all_metrics = {}

        # Вставляем результат для tried_value в список (если не первый шаг)
        if i != 0:
            config_default = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config_default, k, v)
            logging.info(f"[ШАГ {i+1}] {param_name} = {tried_value} (ранее проверенный)")
            dev_mean_default, dev_metrics_default = train_fn(config_default, train_loader, dev_loader, test_loader)
            all_metrics[tried_value] = (dev_mean_default, dev_metrics_default)
            box_text = format_result_box(i+1, param_name, tried_value, {k: v for k, v in current_best_params.items() if k != param_name}, dev_metrics_default, is_best=True)
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")
            best_metric_for_param = dev_mean_default

        for candidate in candidates_to_try:
            config = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config, k, v)
            setattr(config, param_name, candidate)

            logging.info(f"[ШАГ {i+1}] {param_name} = {candidate}, (остальные {current_best_params})")
            dev_mean, dev_metrics = train_fn(config, train_loader, dev_loader, test_loader)

            all_metrics[candidate] = (dev_mean, dev_metrics)

            is_better = dev_mean > best_metric_for_param
            box_text = format_result_box(
                step_num=i+1,
                param_name=param_name,
                candidate=candidate,
                fixed_params={k: v for k, v in current_best_params.items() if k != param_name},
                dev_metrics=dev_metrics,
                is_best=is_better
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")

            if is_better:
                best_val_for_param = candidate
                best_metric_for_param = dev_mean

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n>> [Итог Шаг{i+1}]: Лучший {param_name}={best_val_for_param}, dev_mean={best_metric_for_param:.4f}\n")

    # Итоговая комбинация
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Итоговая комбинация (Dev-based) ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")

    logging.info("Готово! Лучшие параметры подобраны.")
