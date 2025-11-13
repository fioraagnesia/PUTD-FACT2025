import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle

def create_model(data):
    print("\nInside create_model function")
    
    # Select only the features you want to use for training
    selected_features = ["Physical Condition", "Gender", "Breed", "Feed"]
    
    # Print the number of features being used in the model
    print(f"Number of features being used in the model: {len(selected_features)}")
    print(f"Features being used: {selected_features}")
    
    # Separate features (X) and target (y)
    X = data[selected_features]
    y = data["Profit Margin (%)"]

    # Handle missing values in features using SimpleImputer
    imputer = SimpleImputer(strategy="mean")  # Replace NaN values with the mean
    X = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Random Forest Regression
    print("\nApplying Random Forest Regression:")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Regression Results:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    print("R2 Score:", r2_score(y_test, y_pred_rf))

    # Perform Hyperparameter Tuning with GridSearchCV
    print("\nPerforming Hyperparameter Tuning with GridSearchCV for Random Forest:")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                               param_grid=param_grid, 
                               cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    best_rf_model = grid_search.best_estimator_

    # Predict with best model and evaluate
    y_pred_rf_best = best_rf_model.predict(X_test)
    print("Optimized Random Forest Regression Results:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf_best))
    print("R2 Score:", r2_score(y_test, y_pred_rf_best))

    # Perform Cross-Validation for Random Forest
    print("\nPerforming Cross-Validation for Random Forest:")
    cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring="r2")
    print(f"Cross-Validation R2 Scores: {cv_scores}")
    print(f"Mean Cross-Validation R2 Score: {cv_scores.mean()}")

    return best_rf_model, scaler, imputer


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def create_model(data):
    print("\nInside create_model function")
    
    # Select only the 4 features you want to use for training
    selected_features = ["Timestamp (Month)","Timestamp (Year)", "Gender", "Breed",  "Weight (kg)", "Age of Chicken (Months)"]
    
    # Print the number of features being used in the model
    print(f"Number of features being used in the model: {len(selected_features)}")
    print(f"Features being used: {selected_features}")
    
    # Separate features (X) and target (y)
    X = data[selected_features]
    y = data["Price (IDR)"]

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Random Forest Regression
    print("\nApplying Random Forest Regression:")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Regression Results:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    print("R2 Score:", r2_score(y_test, y_pred_rf))

    # Perform Cross-Validation for Random Forest
    print("\nPerforming Cross-Validation for Random Forest:")
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
    print(f"Cross-Validation R2 Scores: {cv_scores}")
    print(f"Mean Cross-Validation R2 Score: {cv_scores.mean()}")

    return rf_model, scaler
def get_clean_data():
    print("\nInside get_clean_data function")

    
    
    # Load dataset
    data = pd.read_csv("data/data.csv")

    print("Columns with missing values and their counts:")
    print(data.isnull().sum())
    
    print(f"Original data shape: {data.shape}")
    
    # Replace invalid placeholders with NaN
    data.replace(['', 'NaN', 'null', '?'], pd.NA, inplace=True)
    
    # Drop rows with missing values
    print("Dropping rows with missing values...")
    data = data.dropna()
    print(f"Data shape after dropping missing values: {data.shape}")
    
    # Drop unnecessary columns
    data = data.drop(columns=["Culture", "Mode of Chick Transport", "Breed_Num", 
                              "Feed_Num", "Culture_Num", "Transport_Num", "Condition_Num", "Profit Margin (%)"], errors='ignore')
    
    # Map categorical data to numerical values
    data['Timestamp (Year)'] = data['Timestamp (Year)'].map({2019: 1, 2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6})
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Breed'] = data['Breed'].map({'Broiler': 2, 'Layer': 1, 'Dual-purpose': 0})
    
    # Print debug info about the dataset
    print(f"Number of rows after cleaning: {data.shape[0]}")
    print(f"Number of columns after cleaning: {data.shape[1]}")
    print(f"Columns after cleaning: {data.columns.tolist()}")

    # Ensure no missing values remain
    print("Final check for missing values:")
    print(data.isnull().sum())
    
    assert data.isnull().sum().sum() == 0, "Data still contains NaN values!"
    
    
    return data

def main():
    print("\nInside main function")  # Debugging line to confirm we're in the main function.
    data = get_clean_data()

    # Unpack the return values from create_model
    rf_model, scaler = create_model(data)

    print("HELLOOOO")  # Debugging line to see if execution reaches here

    # Print debugging to verify models are created
    print("\nSaving models...")

    # Check model type before saving
    print(f"rf_model type: {type(rf_model)}")  # Ensure this is RandomForestRegressor
    print(f"scaler type: {type(scaler)}")  # Ensure this is StandardScaler

    # Save the model and scaler
    try:
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        print("Random Forest model saved successfully.")
    except Exception as e:
        print(f"Error saving Random Forest model: {e}")

    try:
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved successfully.")
    except Exception as e:
        print(f"Error saving scaler: {e}")

if __name__ == '__main__':
    main()
