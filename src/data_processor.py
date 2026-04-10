from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataProcessor:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.label_scalers = {}
        self.label_encoder = LabelEncoder()

    def prepare_features(self, df):
        """Подготовка признаков"""
        # Копируем данные
        features = df.copy()

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
            features['Annealing_atmosphere_encoded'] = self.label_encoder.fit_transform(
                features['Annealing_atmosphere'].fillna('None')
            )

        # Создаем финальный набор признаков
        feature_columns = numerical_features + ['Annealing_atmosphere_encoded']
        X = features[feature_columns]

        return X, feature_columns

    def prepare_targets(self, df):
        """Подготовка целевых переменных"""
        target_columns = [
            'Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
            'Pore_density_pores_per_um2', 'Anatase_ratio'
        ]
        y = df[target_columns]
        return y, target_columns

    def fit_transform(self, train_df):
        """Обучение scalers и преобразование тренировочных данных"""
        X_train, feature_columns = self.prepare_features(train_df)
        y_train, target_columns = self.prepare_targets(train_df)

        # Масштабирование признаков
        X_scaled = self.feature_scaler.fit_transform(X_train)

        # Масштабирование целевых переменных
        y_scaled = {}
        for target in target_columns:
            scaler = StandardScaler()
            y_scaled[target] = scaler.fit_transform(y_train[[target]].values.reshape(-1, 1))
            self.label_scalers[target] = scaler

        return X_scaled, y_scaled, feature_columns, target_columns

    def transform(self, df):
        """Преобразование новых данных"""
        X, feature_columns = self.prepare_features(df)
        y, target_columns = self.prepare_targets(df)

        X_scaled = self.feature_scaler.transform(X)

        y_scaled = {}
        for target in target_columns:
            if target in df.columns:
                y_scaled[target] = self.label_scalers[target].transform(
                    y[[target]].values.reshape(-1, 1)
                )

        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled, target_name):
        """Обратное преобразование целевой переменной"""
        return self.label_scalers[target_name].inverse_transform(y_scaled)

    def save(self, filepath):
        """Сохранение processor"""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """Загрузка processor"""
        return joblib.load(filepath)