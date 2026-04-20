import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

import lime.lime_text


#load
def load_data():
    data = pd.read_csv(r"C:\Users\saibr\OneDrive\Desktop\4th sem\ml\assignments\lab9\fake-news.csv")
    data = data.dropna()
    data['label'] = data['label'].astype(str)
    X = data['text']
    y = data['label']
    return X, y


#train and test split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


#stacking classifer
def create_stacking_model(meta_model=None):
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('dt', DecisionTreeClassifier())
    ]

    if meta_model is None:
        meta_model = LogisticRegression()

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=2)
    )
    return model


def print_stacking_details(model):
    print("A1- Stacking classifier Details")
    print("\nBase Models (Estimators):")
    for name, estimator in model.estimators:
        print(f"  • {name:<5} → {estimator.__class__.__name__}")
    print(f"\nMeta-Model (Final Estimator):")
    print(f"  • {model.final_estimator.__class__.__name__}")
    print(f"\nCross-validation strategy: {model.cv}")



#pipeline
    
def create_pipeline(stacking_model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', stacking_model)
    ])
    return pipeline


def print_pipeline_details(pipeline):
    print("A2 — PIPELINE STEPS")
    for i, (step_name, step_obj) in enumerate(pipeline.steps, 1):
        print(f"  Step {i}: '{step_name}' → {step_obj.__class__.__name__}")


#train
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


#comparision
def compare_metamodels(X_train, X_test, y_train, y_test):
    metamodels = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(),
        'Linear SVC (calibrated)': CalibratedClassifierCV(LinearSVC(max_iter=2000))
    }

    print("A1— Meta Model Comparision (Stacking Final Estimator)")
    print(f"  {'Metamodel':<30} {'Accuracy':>10}")

    best_acc   = -1
    best_name  = None
    best_model = None

    for name, meta in metamodels.items():
        stacking = create_stacking_model(meta_model=meta)
        pipe     = create_pipeline(stacking)
        pipe     = train_model(pipe, X_train, y_train)
        acc      = evaluate_model(pipe, X_test, y_test)
        print(f"  {name:<30} {acc:>10.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = pipe

    print(f"\n Best metamodel: {best_name} (Accuracy = {best_acc:.4f})")
    return best_model


#lime
def explain_with_lime(pipeline, X_test):
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['FAKE', 'REAL']
    )
    exp = explainer.explain_instance(
        X_test.iloc[0],
        pipeline.predict_proba
    )
    return exp


def print_lime_explanation(explanation, pipeline, X_test):
    pred = pipeline.predict([X_test.iloc[0]])[0]

    print("A3 — Lime explainar")
    print(f"\n  Sample text (first 120 chars):")
    print(f"  \"{X_test.iloc[0][:120]}...\"")
    print(f"\n  Prediction: {pred}")
    print("\n  Top words influencing prediction:")
    print(f"  {'Word':<15} {'Weight':>8}   Influence Level")
    print("  " + "-" * 42)

    for word, weight in explanation.as_list():
        influence = "strong" if abs(weight) > 0.3 else "moderate"
        print(f"  {word:<15} {weight:>+8.3f}   ({influence})")

#main
if __name__ == "__main__":

    #Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    #Show stacking model details(default metamodel)
    default_stacking = create_stacking_model()
    print_stacking_details(default_stacking)

    #Show pipeline steps
    sample_pipeline = create_pipeline(default_stacking)
    print_pipeline_details(sample_pipeline)

    #Compare metamodels & pick best
    best_pipeline = compare_metamodels(X_train, X_test, y_train, y_test)

    #LIME explanation on best pipeline
    explanation = explain_with_lime(best_pipeline, X_test)
    print_lime_explanation(explanation, best_pipeline, X_test)
