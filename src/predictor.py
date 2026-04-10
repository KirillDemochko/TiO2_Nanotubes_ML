import pandas as pd
from .data_processor import DataProcessor
from .model_trainer import GPRModelTrainer


class TiO2Predictor:
    def __init__(self, models_directory, processor_path):
        self.models_directory = models_directory
        self.processor_path = processor_path
        self.data_processor = None
        self.model_trainer = None
        self.target_names = [
            'Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
            'Pore_density_pores_per_um2', 'Anatase_ratio'
        ]
        self.is_loaded = False

    def load(self):
        """Загрузка моделей и processor"""
        self.data_processor = DataProcessor.load(self.processor_path)
        self.model_trainer = GPRModelTrainer()
        self.model_trainer.load_models(self.models_directory, self.target_names)
        self.is_loaded = True

    def predict(self, input_data):
        """Предсказание для новых данных"""
        if not self.is_loaded:
            self.load()

        # Преобразование входных данных
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input data should be dict or DataFrame")

        X_scaled, _ = self.data_processor.transform(input_df)

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

    def predict_batch(self, input_df):
        """Пакетное предсказание"""
        if not self.is_loaded:
            self.load()

        X_scaled, _ = self.data_processor.transform(input_df)

        results = []
        for i in range(len(input_df)):
            row_predictions = {}
            row_uncertainties = {}

            for target_name in self.target_names:
                # Для пакетного предсказания используем только одну точку
                X_single = X_scaled[i:i + 1]
                y_pred, y_std = self.model_trainer.predict(X_single, target_name, return_std=True)

                y_pred_original = self.data_processor.inverse_transform_target(
                    y_pred.reshape(-1, 1), target_name
                ).flatten()[0]

                y_std_original = y_std[0] * self.data_processor.label_scalers[target_name].scale_

                row_predictions[target_name] = y_pred_original
                row_uncertainties[target_name] = y_std_original

            results.append({
                'predictions': row_predictions,
                'uncertainties': row_uncertainties
            })

        return results