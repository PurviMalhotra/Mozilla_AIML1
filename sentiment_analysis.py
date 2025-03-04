import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):

    if file_path:
        try:
            df = pd.read_csv('/Users/purvimalhotra/Developer/ccode/MFtask1/Data.csv')
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using example data instead.")
    
    data = {
        'Sentence': [
            "The GeoSolutions technology will leverage Benefon 's GPS solutions by providing Location Based Search Technology , a Communities Platform , location relevant multimedia content and a new and powerful co",
            "$ESI on lows, down $1.50 to $2.50 BK a real possibility",
            "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .",
            "According to the Finnish-Russian Chamber of Commerce , all the major construction companies of Finland are operating in Russia .",
            "The Swedish buyout firm has sold its remaining 22.4 percent stake , almost eighteen months after taking the company public in Finland .",
            "$SPY wouldn't be surprised to see a green close",
            "Shell's $70 Billion BG Deal Meets Shareholder Skepticism",
            "SSH COMMUNICATIONS SECURITY CORP STOCK EXCHANGE RELEASE OCTOBER 14 , 2008 AT 2:45 PM The Company updates its full year outlook and estimates its results to remain at loss for the full year .",
            "Kone 's net sales rose by some 14 % year-on-year in the first nine months of 2008 ."
        ],
        'Sentiment': [
            "positive", "negative", "positive", "neutral", "neutral", 
            "positive", "negative", "negative", "positive"
        ]
    }
    return pd.DataFrame(data)

#preprocessing text
def preprocess_text(df):

    processed_df = df.copy()
    
    if 'Sentence' in processed_df.columns and 'text' not in processed_df.columns:
        processed_df = processed_df.rename(columns={'Sentence': 'text'})
    if 'Sentiment' in processed_df.columns and 'sentiment' not in processed_df.columns:
        processed_df = processed_df.rename(columns={'Sentiment': 'sentiment'})
    
    processed_df['text'] = processed_df['text'].fillna("")
    
    return processed_df

#train the model with hyperparameter tuning
def train_sentiment_model(data, test_size=0.2, random_state=42, perform_grid_search=True):

    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], 
        data['sentiment'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=data['sentiment'] if len(data) > 10 else None
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    #pipeline with TF-IDF vectorizer and SVM classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
        ('svm', LinearSVC(class_weight='balanced', max_iter=10000))
    ])
    
    #grid search
    param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'svm__C': [0.1, 1, 10],
        }
        
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
        
    print(f"Best parameters: {grid_search.best_params_}")
    pipeline = grid_search.best_estimator_
    
    #eval on test data
    y_pred = pipeline.predict(X_test)
    
    #calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test, y_pred, conf_matrix, report

#visualise confusing matrics
def plot_confusion_matrix(conf_matrix, class_names):
    #heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

#predict sentiment on new
def predict_sentiment(model, texts):
    #model: trained pipeline
    #texts: list of strings to analyze
    predictions = model.predict(texts)
    return pd.DataFrame({'text': texts, 'predicted_sentiment': predictions})

#feature importance
def analyze_feature_importance(model, class_names, top_n=20):
    #class_names: list of sentiment classes
    #top_n: number of top features to display
    
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    
    coefficients = model.named_steps['svm'].coef_
    
    for i, class_name in enumerate(class_names):
        if coefficients.shape[0] > i:  
            
            top_positive_indices = np.argsort(coefficients[i])[-top_n:]
            top_positive_features = [(feature_names[j], coefficients[i][j]) for j in top_positive_indices]
            
            print(f"\nTop {top_n} features for class '{class_name}':")
            for feature, coef in reversed(top_positive_features):
                print(f"  {feature}: {coef:.4f}")

#main
def main(file_path=None):
    
    data = load_data(file_path)
    
    processed_data = preprocess_text(data)
    
    print("\nClass distribution:")
    print(processed_data['sentiment'].value_counts())
    
    model, X_test, y_test, y_pred, conf_matrix, report = train_sentiment_model(processed_data)
    
    class_names = sorted(processed_data['sentiment'].unique())
    plot_confusion_matrix(conf_matrix, class_names)
    
    analyze_feature_importance(model, class_names)
    
    #example
    new_texts = [
        "Today was a terrible day for many due to the california fires.",
        "Many still wonder if the convict was actually the culprit.",
        "I finally finished my task for mozilla firefox club recruitments."
    ]
    
    predictions = predict_sentiment(model, new_texts)
    print("\nSentiment predictions for new texts:")
    print(predictions)
    
    return model, report

if __name__ == "__main__":
    model, metrics = main('/Users/purvimalhotra/Developer/ccode/MFtask1/Data.csv')
    
    
