import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('highest_grossing_movies.csv')

# Display the first few rows of the dataset
print(df.head())

# Basic information about the dataset
print(df.info())

# Exploratory Data Analysis (EDA)

# 1. Box Office Trends Over the Years
def plot_box_office_trends(df):
    plt.figure(figsize=(12, 6))
    yearly_revenue = df.groupby('year')['box_office'].sum().reset_index()
    sns.lineplot(data=yearly_revenue, x='year', y='box_office')
    plt.title('Total Box Office Revenue Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Box Office Revenue (in billions)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

plot_box_office_trends(df)

# 2. Genre Analysis
def plot_genre_distribution(df):
    plt.figure(figsize=(12, 6))
    genres = df['genre'].str.split(',', expand=True).stack().value_counts()
    sns.barplot(x=genres.values, y=genres.index)
    plt.title('Distribution of Genres in Highest-Grossing Movies')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genres')
    plt.grid()
    plt.show()

plot_genre_distribution(df)

# 3. Cast and Crew Influence on Box Office Performance
def analyze_cast_crew_impact(df):
    plt.figure(figsize=(12, 6))
    top_actors = df['lead_actor'].value_counts().head(10).index
    top_movies = df[df['lead_actor'].isin(top_actors)]
    sns.boxplot(data=top_movies, x='lead_actor', y='box_office')
    plt.title('Box Office Performance of Top Actors')
    plt.xlabel('Lead Actor')
    plt.ylabel('Box Office Revenue (in billions)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

analyze_cast_crew_impact(df)

# 4. Production Budget vs. Box Office Revenue
def plot_budget_vs_revenue(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='production_budget', y='box_office')
    plt.title('Production Budget vs. Box Office Revenue')
    plt.xlabel('Production Budget (in millions)')
    plt.ylabel('Box Office Revenue (in billions)')
    plt.grid()
    plt.show()

plot_budget_vs_revenue(df)

# 5. Box Office Revenue Prediction - Example of a Simple Model (optional)
def box_office_revenue_prediction(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Select relevant features for the model
    X = df[['production_budget']]
    y = df['box_office']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Plot the predicted vs. actual box office revenues
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('Predicted vs. Actual Box Office Revenue')
    plt.xlabel('Actual Box Office Revenue (in billions)')
    plt.ylabel('Predicted Box Office Revenue (in billions)')
    plt.grid()
    plt.show()

box_office_revenue_prediction(df)
