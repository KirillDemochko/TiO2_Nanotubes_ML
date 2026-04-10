from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import joblib


class GPRModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}

    def create_kernel(self):
        """Создание ядра для GPR"""
        # Комбинация различных ядер для лучшей производительности
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-5)
        return kernel
    def train_models(self, X_train, y_train_dict, target_names):
        """Обучение отдельных моделей для каждой целевой переменной"""
        for target_name in target_names:
            print(f"Training model for {target_name}...")

            y_train = y_train_dict[target_name].ravel()

            # Создание и обучение модели
            kernel = self.create_kernel()
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-5,
                n_restarts_optimizer=10,
                random_state=self.random_state
            )

            gpr.fit(X_train, y_train)
            self.models[target_name] = gpr

            print(f"Model for {target_name} trained. Kernel: {gpr.kernel_}")

        return self.models

    def predict(self, X, target_name, return_std=True):
        """Предсказание для конкретной целевой переменной"""
        if target_name not in self.models:
            raise ValueError(f"Model for {target_name} not found")

        model = self.models[target_name]
        if return_std:
            return model.predict(X, return_std=return_std)
        else:
            return model.predict(X)

    def save_models(self, directory):
        """Сохранение обученных моделей"""
        import os
        os.makedirs(directory, exist_ok=True)

        for target_name, model in self.models.items():
            filename = f"{directory}/{target_name}.pkl"
            joblib.dump(model, filename)

    def load_models(self, directory, target_names):
        """Загрузка обученных моделей"""
        for target_name in target_names:
            filename = f"{directory}/{target_name}.pkl"
            self.models[target_name] = joblib.load(filename)