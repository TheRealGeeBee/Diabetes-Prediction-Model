class DiabetesModel:
    def __init__(self, df):
        self.df = df

    def scale_data(self):
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        import numpy as np

        new_data = np.reshape(self.df, (-1, 3))
        pf = PolynomialFeatures(degree=3)
        new_data_transformed = pf.fit_transform(new_data)
        ss = StandardScaler()
        ss.fit_transform(new_data_transformed)
        return new_data_transformed

    def predict_value(self):
        import joblib as jb
        pred_float = 0.00
        model = jb.load(r"C:\Users\Gabi\Desktop\Trained Machine Learning Models\Diabetes Prediction Model.pkl")
        data = self.scale_data()
        pred = model.predict(data)
        for items in pred:
            pred_float = items
        print(f"This person has a {round(pred_float*100, 2)}% chance of being diagnosed diabetic")

new_person = DiabetesModel([93, 405, 1])
new_person.predict_value()