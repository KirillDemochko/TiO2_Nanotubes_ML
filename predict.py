import pandas as pd
import sys
import os
from data_processor import DataProcessor
from model_trainer import GPRModelTrainer


class TiO2InteractivePredictor:
    def __init__(self, models_directory='models', processor_path='models/feature_scaler.pkl'):
        self.models_directory = models_directory
        self.processor_path = processor_path
        self.data_processor = None
        self.model_trainer = None
        self.target_names = [
            'Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
            'Pore_density_pores_per_um2', 'Anatase_ratio'
        ]
        self.feature_names = [
            'Ethylene_glycol_vol_percent', 'Water_vol_percent', 'NH4F_wt_percent',
            'Glycerol_vol_percent', 'HF_vol_percent', 'Voltage_V',
            'Anodization_time_min', 'Temperature_anodization_C',
            'Annealing_temperature_C', 'Annealing_time_min',
            'Substrate_thickness_mm', 'Substrate_area_cm2', 'Annealing_atmosphere'
        ]
        self.is_loaded = False

    def load_models(self):
        """Загрузка моделей и processor"""
        if not self.is_loaded:
            try:
                # Проверяем существование файлов
                if not os.path.exists(self.processor_path):
                    raise FileNotFoundError(f"Processor file not found: {self.processor_path}")

                # Проверяем существование моделей
                for target_name in self.target_names:
                    model_path = f"{self.models_directory}/{target_name}.pkl"
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found: {model_path}")

                self.data_processor = DataProcessor.load(self.processor_path)
                self.model_trainer = GPRModelTrainer()
                self.model_trainer.load_models(self.models_directory, self.target_names)
                self.is_loaded = True
                print("Модели успешно загружены!")
            except Exception as e:
                print(f"Ошибка загрузки моделей: {e}")
                print("Убедитесь, что модели обучены и находятся в папке 'models/'")
                sys.exit(1)

    def show_input_instructions(self, mode):
        """Показать инструкции по вводу данных"""
        print("\n" + "=" * 50)
        if mode == 1:
            print("РЕЖИМ ПРЕДСКАЗАНИЯ")
            print("Введите 13 параметров синтеза через запятую:")
        else:
            print("РЕЖИМ ПРЕДСКАЗАНИЯ И ОЦЕНКИ")
            print("Введите 13 параметров синтеза + 5 экспериментальных результатов через запятую:")

        print("\nПараметры синтеза (13):")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feature}")

        if mode == 2:
            print("\nЭкспериментальные результаты (5):")
            for i, target in enumerate(self.target_names, 14):
                print(f"  {i:2d}. {target}")

        print("\nПример ввода для режима предсказания:")
        print("50.0, 30.0, 0.5, 15.0, 2.0, 40.0, 120.0, 25.0, 450.0, 60.0, 0.2, 5.0, Air")

        if mode == 2:
            print("\nПример ввода для режима предсказания и оценки:")
            print(
                "50.0, 30.0, 0.5, 15.0, 2.0, 40.0, 120.0, 25.0, 450.0, 60.0, 0.2, 5.0, Air, 45.0, 18.0, 25.0, 350.0, 0.3")

        print("=" * 50)

    def parse_input(self, input_string, mode):
        """Разбор введенной строки"""
        try:
            values = [x.strip() for x in input_string.split(',')]

            if mode == 1 and len(values) != 13:
                raise ValueError(f"Ожидается 13 параметров, получено {len(values)}")
            elif mode == 2 and len(values) != 18:
                raise ValueError(f"Ожидается 18 параметров (13+5), получено {len(values)}")

            # Преобразование числовых значений
            synthesis_params = {}
            for i, feature in enumerate(self.feature_names):
                if i < 12:  # Числовые параметры
                    try:
                        synthesis_params[feature] = float(values[i])
                    except ValueError:
                        raise ValueError(f"Параметр '{feature}' должен быть числом, получено: '{values[i]}'")
                else:  # Annealing_atmosphere (строковый)
                    synthesis_params[feature] = values[i]

            if mode == 2:
                experimental_results = {}
                for i, target in enumerate(self.target_names):
                    try:
                        experimental_results[target] = float(values[13 + i])
                    except ValueError:
                        raise ValueError(f"Результат '{target}' должен быть числом, получено: '{values[13 + i]}'")
                return synthesis_params, experimental_results
            else:
                return synthesis_params, None

        except ValueError as e:
            print(f"Ошибка разбора ввода: {e}")
            return None, None

    def prepare_features_for_prediction(self, synthesis_params):
        """Подготовка признаков для предсказания (без целевых переменных)"""
        # Копируем данные
        features = pd.DataFrame([synthesis_params])

        # Выделяем числовые признаки
        numerical_features = [
            'Ethylene_glycol_vol_percent', 'Water_vol_percent',
            'NH4F_wt_percent', 'Glycerol_vol_percent', 'HF_vol_percent',
            'Voltage_V', 'Anodization_time_min', 'Temperature_anodization_C',
            'Annealing_temperature_C', 'Annealing_time_min',
            'Substrate_thickness_mm', 'Substrate_area_cm2'
        ]

        # Кодируем категориальную переменную
        if 'Annealing_atmosphere' in features.columns:
            # Используем обученный LabelEncoder из data_processor
            features['Annealing_atmosphere_encoded'] = self.data_processor.label_encoder.transform(
                features['Annealing_atmosphere'].fillna('None')
            )

        # Создаем финальный набор признаков
        feature_columns = numerical_features + ['Annealing_atmosphere_encoded']
        X = features[feature_columns]

        # Масштабирование признаков
        X_scaled = self.data_processor.feature_scaler.transform(X)

        return X_scaled

    def predict_from_input(self, synthesis_params):
        """Предсказание на основе параметров синтеза"""
        if not self.is_loaded:
            self.load_models()

        try:
            # Подготовка признаков
            X_scaled = self.prepare_features_for_prediction(synthesis_params)

            # Предсказание для всех целевых переменных
            predictions = {}
            uncertainties = {}

            for target_name in self.target_names:
                y_pred, y_std = self.model_trainer.predict(X_scaled, target_name, return_std=True)

                # Обратное преобразование к оригинальному масштабу
                y_pred_original = self.data_processor.inverse_transform_target(
                    y_pred.reshape(-1, 1), target_name
                ).flatten()

                y_std_original = y_std * self.data_processor.label_scalers[target_name].scale_

                predictions[target_name] = y_pred_original[0]
                uncertainties[target_name] = y_std_original[0]

            return predictions, uncertainties

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None, None

    def calculate_metrics(self, predictions, experimental_results):
        """Расчет метрик качества"""
        metrics = {}
        for target_name in self.target_names:
            true_val = experimental_results[target_name]
            pred_val = predictions[target_name]

            absolute_error = abs(true_val - pred_val)
            relative_error = (absolute_error / true_val) * 100 if true_val != 0 else float('inf')

            metrics[target_name] = {
                'Экспериментальное значение': true_val,
                'Предсказанное значение': pred_val,
                'Абсолютная ошибка': absolute_error,
                'Относительная ошибка (%)': relative_error
            }

        return metrics

    def format_output(self, predictions, uncertainties, metrics=None):
        print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
        print("=" * 60)

        for target_name in self.target_names:
            print(f"\n{target_name}:")
            print(f"  Предсказанное значение: {predictions[target_name]:.4f}")
            print(f"  Неопределенность: ±{uncertainties[target_name]:.4f}")

            if metrics and target_name in metrics:
                metric = metrics[target_name]
                print(f"  Экспериментальное значение: {metric['Экспериментальное значение']:.4f}")
                print(f"  Абсолютная ошибка: {metric['Абсолютная ошибка']:.4f}")
                print(f"  Относительная ошибка: {metric['Относительная ошибка (%)']:.2f}%")


    def run_interactive_mode(self):
        """Запуск интерактивного режима"""
        print("TiO2 Nanotube Properties Predictor")
        print("=" * 40)

        while True:
            print("\nВыберите режим работы:")
            print("1 - Только предсказание (13 параметров)")
            print("2 - Предсказание и оценка (18 параметров)")
            print("0 - Выход")

            try:
                choice = input("\nВаш выбор (0-2): ").strip()

                if choice == '0':
                    print("До свидания!")
                    break
                elif choice in ['1', '2']:
                    mode = int(choice)
                    self.process_mode(mode)
                else:
                    print("Неверный выбор. Попробуйте снова.")
            except KeyboardInterrupt:
                print("\nПрограмма завершена.")
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    def process_mode(self, mode):
        """Обработка выбранного режима"""
        self.show_input_instructions(mode)

        input_string = input("\nВведите данные через запятую: ")

        synthesis_params, experimental_results = self.parse_input(input_string, mode)

        if synthesis_params is None:
            return

        if mode == 1:
            predictions, uncertainties = self.predict_from_input(synthesis_params)
            if predictions is not None:
                self.format_output(predictions, uncertainties)
        else:
            if experimental_results is not None:
                predictions, uncertainties = self.predict_from_input(synthesis_params)
                if predictions is not None:
                    metrics = self.calculate_metrics(predictions, experimental_results)
                    self.format_output(predictions, uncertainties, metrics)


# Функция для запуска из командной строки
def main():
    predictor = TiO2InteractivePredictor()
    predictor.run_interactive_mode()


if __name__ == "__main__":
    main()