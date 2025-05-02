from src.data_loader import load_data, explore_data
from src.preprocesss import apply_preprocessing
from src.feature_extraction import extract_features
from src.model_traning import train_model
from src.evaluation import evaluate_model

from sklearn.model_selection import train_test_split
def main():
    # 1. Load & explore
    df = load_data('data/spam_or_not_spam_data.csv')
    explore_data(df)

    # Preprocessing (adds 'clean_email')
    df = apply_preprocessing(df, 'email')

    # Drop any rows where cleaning left an empty text
    df = df.dropna(subset=['clean_email'])

    # Feature extraction: unpack both outputs
    X, vectorizer = extract_features(df, text_column='clean_email')
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #  Train and evaluate
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
