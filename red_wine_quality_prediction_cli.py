import pickle
from typing import List


class XGBoostRegressionModel:

    def __init__(self, model_pickle_path: str):
        self.model_pickle_path = model_pickle_path
        self.model = self.load_model()

    def load_model(self):
        with open(self.model_pickle_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        return model

    def predict(self, feature_list: List[float]):
        return self.model.predict(feature_list)


def user_interaction(model: XGBoostRegressionModel):
    print('Please enter the following characteristics:')
    fixed_acidity = float(input('Fixed Acidity: '))
    volatile_acidity = float(input('Volatile Acidity: '))
    citric_acid = float(input('Citric Acid: '))
    residual_sugar = float(input('Residual Sugar: '))
    chlorides = float(input('Chlorides: '))
    free_sulfur_dioxide = float(input('Free Sulfur Dioxide: '))
    total_sulfur_dioxide = float(input('Total Sulfur Dioxide: '))
    density = float(input('Density: '))
    ph = float(input('pH: '))
    sulphates = float(input('Sulphates '))
    alcohol = float(input('Alcohol: '))

    features_list = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol
    ]]

    quality = model.predict(features_list)

    print(f'\nYour wine quality is: {quality[0]:.1f}')


def main():
    print('--- Wine Quality Prediction ---')
    model = XGBoostRegressionModel(model_pickle_path='model/xgb_reg.pickle')

    while True:
        user_interaction(model)

        user_choice = input('\nDo you want to enter another wine? (Y/n): ')

        if user_choice == 'n':
            break


if __name__ == '__main__':
    main()
