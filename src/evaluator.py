import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def calculate_metrics(self, y_true, y_pred, target_name):
        """Расчет метрик качества"""
        y_true_original = self.data_processor.inverse_transform_target(
            y_true.reshape(-1, 1), target_name
        ).flatten()

        y_pred_original = self.data_processor.inverse_transform_target(
            y_pred.reshape(-1, 1), target_name
        ).flatten()

        r2 = r2_score(y_true_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        mae = mean_absolute_error(y_true_original, y_pred_original)

        # Процент ошибки (MAPE)
        mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100

        return {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape
        }

    def evaluate_models(self, X_test, y_test_dict, model_trainer, target_names):
        """Оценка всех моделей"""
        results = {}

        for target_name in target_names:
            print(f"\nEvaluating {target_name}:")

            # Предсказание
            y_pred, y_std = model_trainer.predict(X_test, target_name, return_std=True)
            y_test = y_test_dict[target_name].flatten()

            # Расчет метрик
            metrics = self.calculate_metrics(y_test, y_pred, target_name)
            results[target_name] = metrics

            # Вывод результатов
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

        return results

    def plot_predictions(self, X_test, y_test_dict, model_trainer, target_names, n_plots=5):
        """Визуализация предсказаний"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, target_name in enumerate(target_names[:n_plots]):
            y_pred, y_std = model_trainer.predict(X_test, target_name, return_std=True)
            y_test = y_test_dict[target_name].flatten()

            # Обратное преобразование к оригинальному масштабу
            y_test_orig = self.data_processor.inverse_transform_target(
                y_test.reshape(-1, 1), target_name
            ).flatten()

            y_pred_orig = self.data_processor.inverse_transform_target(
                y_pred.reshape(-1, 1), target_name
            ).flatten()

            # График предсказаний vs истинных значений
            axes[i].scatter(y_test_orig, y_pred_orig, alpha=0.6)
            axes[i].plot([y_test_orig.min(), y_test_orig.max()],
                         [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predictions')
            axes[i].set_title(f'{target_name}\nR² = {r2_score(y_test_orig, y_pred_orig):.3f}')

        # Скрываем лишние subplots
        for i in range(len(target_names), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()