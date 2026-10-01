import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("weatherAUS.csv").dropna(subset=["RainTomorrow"])
X = df.drop(columns="RainTomorrow")
y = (df["RainTomorrow"] == "Yes").astype(int)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
# add: from sklearn.impute import SimpleImputer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("prep", preprocess),
                 ("clf", RandomForestClassifier(random_state=42))])

grid = GridSearchCV(
    pipe,
    {"clf__n_estimators": [50, 100],
     "clf__max_depth": [None, 10, 20],
     "clf__class_weight": [None, "balanced"]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1)

grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
