import pandas as pd
import os
from src.data_processor import DataProcessor
from src.model_trainer import GPRModelTrainer
from src.evaluator import ModelEvaluator


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('src', exist_ok=True)

    # Загрузка данных
    print("Loading data...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')

    # Подготовка данных
    print("Preparing data...")
    processor = DataProcessor()
    X_train, y_train_dict, feature_columns, target_names = processor.fit_transform(train_df)
    X_test, y_test_dict = processor.transform(test_df)

    print(f"Features: {feature_columns}")
    print(f"Targets: {target_names}")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Обучение моделей
    print("\nTraining models...")
    trainer = GPRModelTrainer()
    trainer.train_models(X_train, y_train_dict, target_names)

    # Оценка моделей
    print("\nEvaluating models...")
    evaluator = ModelEvaluator(processor)
    results = evaluator.evaluate_models(X_test, y_test_dict, trainer, target_names)

    # Визуализация результатов
    print("\nGenerating plots...")
    evaluator.plot_predictions(X_test, y_test_dict, trainer, target_names)

    # Сохранение моделей и processor
    print("\nSaving models...")
    trainer.save_models('models')
    processor.save('models/feature_scaler.pkl')

    # Вывод суммарных результатов
    print("\nFINAL RESULTS")
    for target_name, metrics in results.items():
        print(f"\n{target_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()