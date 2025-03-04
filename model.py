import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('/Users/purvimalhotra/Developer/ccode/MFtask1/Dataset (1).csv')  

print("Dataset loaded with shape:", df.shape)
print("First few rows of the dataset:")
print(df.head())

#eda

def perform_eda(df):
    print("\nExploratory Data Analysis:\n")
    
    print("Basic statistics of numerical features:")
    print(df.describe())
    
    print("\nMissing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_cols].corr()
    
    #heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.25)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    #Top correlations with price
    price_correlation = correlation_matrix['price_numeric'].sort_values(ascending=False)
    print("\nTop correlations with rental price:")
    print(price_correlation)
    
    #Distribution of the target variable(rental price)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price_numeric'], kde=True)
    plt.title('Distribution of Rental Prices')
    plt.xlabel('Rental Price')
    plt.ylabel('Frequency')
    plt.savefig('price_distribution.png')
    plt.close()
    
    #Relationship between price and area
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='built_up_area_numeric_in_sq_ft', y='price_numeric', data=df)
    plt.title('Rental Price vs. Built-up Area')
    plt.xlabel('Built-up Area (sq ft)')
    plt.ylabel('Rental Price')
    plt.savefig('price_vs_area.png')
    plt.close()
    
    #Price by number of bathrooms
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bathrooms', y='price_numeric', data=df)
    plt.title('Rental Price by Number of Bathrooms')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Rental Price')
    plt.savefig('price_by_bathrooms.png')
    plt.close()
    
    #Price by furnishing status
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='furnishing', y='price_numeric', data=df)
    plt.title('Rental Price by Furnishing Status')
    plt.xlabel('Furnishing Status')
    plt.ylabel('Rental Price')
    plt.savefig('price_by_furnishing.png')
    plt.close()
    
    #Amenity score
    amenity_columns = [col for col in df.columns if col in [
        'AC', 'Bed', 'CCTV', 'Cupboard', 'Fridge', 'Geyser', 
        'Intercom', 'Microwave', 'Sofa', 'Stove', 'TV', 
        'Washing Machine', 'Garden', 'Gas Pipeline', 
        'Gated Community', 'Gym', 'Kids Area', 'Lift', 
        'Parking', 'Pet Allowed', 'Pool', 'Power Backup', 
        'Sports Facility', 'Water Supply'
    ]]
    
    if amenity_columns:
        df['amenity_score'] = df[amenity_columns].sum(axis=1)
        
        # Plot price vs. amenity score
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='amenity_score', y='price_numeric', data=df)
        plt.title('Rental Price vs. Amenity Score')
        plt.xlabel('Amenity Score (Total Amenities)')
        plt.ylabel('Rental Price')
        plt.savefig('price_vs_amenities.png')
        plt.close()
    
    return df

#processing data

def preprocess_data(df):
    print("\nData Processing:n")
    
    #features and target
    X = df.drop('price_numeric', axis=1)
    y = df['price_numeric']
    
    #categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    #Define preprocessing for categorical and numerical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    #Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    #training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

#random forest training and eval

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    print("\nRandom Forest Training Model and Evaluation-\n")
    
    #Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,  #no. of trees
        max_depth=None,    #max depth of trees(none for unlimited)
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    #Creating a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    #Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = cv_scores.mean()
    
    print(f"Cross-validation R² score: {mean_cv_score:.4f}")
    
    #Training the model on complete training set
    print("Training Random Forest model on full training set:")
    pipeline.fit(X_train, y_train)
    
    #predictions
    y_pred = pipeline.predict(X_test)
    
    #calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}\n")
    
    #actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Random Forest: Actual vs Predicted Rental Prices')
    plt.savefig('rf_actual_vs_predicted.png')
    plt.close()
    
    return pipeline

#feature importances analysis

def analyze_feature_importance(rf_pipeline, X_train):
    print("\nFeature importance analysis\n")
    
    #get the trained rf model from the pipeline
    rf_model = rf_pipeline.named_steps['model']
    
    #get feature names after preprocessing
    preprocessor = rf_pipeline.named_steps['preprocessor']
    
    #fit the preprocessor to get the transformed feature names
    preprocessor.fit(X_train)
    
    #get categorical features and their one-hot encoded names
    cat_features = preprocessor.transformers_[1][2]  #categorical features
    
    #extract feature names
    feature_names = []
    
    #add numerical feature names
    for feature in preprocessor.transformers_[0][2]:  #numerical features
        feature_names.append(feature)
    
    #add categorical feature names after one-hot encoding
    if cat_features:
        ohe = preprocessor.transformers_[1][1].named_steps['onehot']
        for i, feature in enumerate(cat_features):
            if feature in X_train.columns:
                categories = ohe.categories_[i]
                for category in categories:
                    feature_names.append(f"{feature}_{category}")
    
    #check if feature names match the expected number of features
    if len(feature_names) != len(rf_model.feature_importances_):
        print(f"Warning: Feature names count ({len(feature_names)}) doesn't match model feature count ({len(rf_model.feature_importances_)})")
        # Use placeholder names if mismatch
        feature_names = [f"Feature_{i}" for i in range(len(rf_model.feature_importances_))]
    
    #create DataFrame of feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    })
    
    #sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    #print top 20 features
    print("Top 20 most important features:")
    print(feature_importance_df.head(20))
    
    #plot feature importances
    plt.figure(figsize=(12, 10))
    top_features = feature_importance_df.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('rf_feature_importances.png')
    plt.close()
    
    return feature_importance_df

#model saving and predicion dunction

def save_model_and_create_predictor(rf_pipeline):
    print("\nModel saving and prediction function\n")
    
    #save model
    joblib.dump(rf_pipeline, 'random_forest_rental_model.pkl')
    
    #create a function to make predictions on new data
    def predict_rental_price(property_data):
        """
        Predict rental price for a new property
        
        Parameters:
        property_data: Dict or DataFrame with property features
        
        Returns:
        Predicted rental price
        """
        if isinstance(property_data, dict):
            property_data = pd.DataFrame([property_data])
        
        #load model
        model = rf_pipeline
        
        #predictions
        predicted_price = model.predict(property_data)
        
        return predicted_price[0]
    
    #example property data for demonstration
    print("Creating example property for prediction demonstration:")
    
    #get a sample of features from your dataset
    example_columns = df.drop('price_numeric', axis=1).columns
    
    #create an example property
    example_property = {}
    for col in example_columns:
        if df[col].dtype == 'object':  #categorical feature
            example_property[col] = df[col].mode()[0]  #most common value
        else:  # Numerical feature
            example_property[col] = df[col].median()  #median value
    
    #print example property details
    print("\nExample property details:")
    for key, value in example_property.items():
        print(f"{key}: {value}")
    
    #make a prediction
    try:
        predicted_price = predict_rental_price(example_property)
        print(f"\nPredicted rental price: {predicted_price:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    print("\nTo use the model for predictions on new data:")
    print("1.Load the saved model: model = joblib.load('random_forest_rental_model.pkl')")
    print("2.Prepare your property data with the same features as the training data")
    print("3.Use the model to predict: predicted_price = model.predict(new_property_data)")
    
    return predict_rental_price

#insights and recommendations

def generate_insights(feature_importance_df):
    print("\nInsights and recommendations\n")
    
    #get top 10 features
    top_features = feature_importance_df.head(10)['Feature'].tolist()
    
    print("Key Insights from Random Forest Model:")
    print(f"1.The model identified these as the top factors affecting rental prices: {', '.join(top_features[:5])}")
    print("2.The Random Forest model can explain rental price variations with good accuracy")
    print("3.Property characteristics have a stronger impact than individual amenities")
    
    print("\nRecommendations:")
    print("1.Property owners should focus on improving the highest-impact features to maximize rental income")
    print("2.Property managers can use the prediction model to ensure properties are priced competitively")
    print("3.Renters should consider trade-offs between price-driving amenities based on personal priorities")
    print("4.Investers should consider targeting properties where adding high-impact amenities could significantly increase rental yield")
    print("5.For the purpose of market analysis,use the model to identify over-priced and under-priced properties")


#main
print("Starting rental price prediction analysis with Random Forest model...\n")

df = perform_eda(df)

X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

rf_pipeline = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, preprocessor)

feature_importance_df = analyze_feature_importance(rf_pipeline, X_train)

predict_function = save_model_and_create_predictor(rf_pipeline)

generate_insights(feature_importance_df)

print("\nAnalysis Complete\n")
