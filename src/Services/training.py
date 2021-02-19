class TrainingService:
    def __init__(self, data_train, target_name, preprocessor=None, model=None) -> None:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        self.data_train = data_train
        self.target_name = target_name
        self.X = self.data_train.drop(columns=self.target_name)
        self.y = self.data_train[self.target_name].copy()
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, train_size=0.75, test_size=0.25, random_state=42)
        if preprocessor is None:
            self.preprocessor = self.get_preprocessor()
        else:
            self.preprocessor = preprocessor
        if model is None:
            self.model = LinearRegression()
        else:
            self.model = model

    def train(self):
        from sklearn.pipeline import Pipeline
        self.clf = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
        self.clf.fit(self.X_train, self.y_train)

    def save_clf(self, name_file=None):
        import pickle
        name_file = 'outputs/classifier.pkl' if name_file is None else 'outputs/' + \
            name_file + '.pkl'
        try:
            with open(name_file, 'wb') as pkl_file:
                pickle.dump(self.clf, pkl_file)
        except:
            print(
                "Cannot save a classifier from TrainService, run TrainService.train() before trying to save it.")

    def get_preprocessor(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        numerical_cols = self.X_train.drop(
            columns='Id').select_dtypes(exclude='object').columns
        categorical_cols = self.X_train.select_dtypes(include='object').columns

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )

        return preprocessor
