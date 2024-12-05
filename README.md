import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Assume we have a dataset 'phishing_emails.csv' with two columns 'email' and 'label'
# 'label' column contains 0 for non-phishing and 1 for phishing emails

data = pd.read_csv('phishing_emails.csv')

# Preview the data
print(data.head())

# Step 1: Preprocessing the email text data
# We will remove any null values and convert everything to lowercase
data['email'] = data['email'].str.lower()

# Step 2: Feature extraction using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['email'])

# Step 3: Prepare target variable 'y' and split the dataset into train and test sets
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n {classification_report(y_test, y_pred)}')

# Example of predicting a new email
new_email = ["Congratulations! You've won a $1000 gift card! Click here to claim it."]
new_email_transformed = tfidf_vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)

if prediction == 1:
    print("The email is phishing!")
else:
    print("The email is not phishing.")

