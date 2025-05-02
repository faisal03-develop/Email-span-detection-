from src.data_loader import load_data
from src.preprocesss import apply_preprocessing
from src.feature_extraction import extract_features
from src.model_traning import train_model
from src.evaluation import evaluate_model
from src.threshold_tunning import tune_threshold

def main():
    print("\n Load data")
    df = load_data('data/spam_or_not_spam_data.csv')

    print("\n preprcess emails..")
    df = apply_preprocessing(df, text_column='email')

    print("\n extract features")
    X, vectorizer = extract_features(df, text_column='clean_email')

    y = df['label']

    print("\n split data into train and test sets")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n Train model")
    model = train_model(X_train, y_train)

    print("\n model performance.")
    evaluate_model(model, X_test, y_test)

    print("\n Tuning ")
    tune_threshold(model, X_test, y_test)

if __name__ == "__main__":
    main()
