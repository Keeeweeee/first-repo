import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = {
    "Messages": [
        "Win money now!!!", "Lowest price on meds, buy today",
        "Congratulations, you won a lottery", "Get cheap loans approved instantly",
        "Earn cash fast, click this link", "Winner! Claim your prize now",
        "Special discount just for you", "You have been selected for a reward",
        "Act now for free vacation", "Exclusive offer limited time",
        "Hi, how are you?", "Are we still meeting tomorrow?",
        "Can we call later?", "Dinner at my place tonight?",
        "Don’t forget the meeting at 3", "Happy birthday! Wishing you the best",
        "Let’s grab coffee tomorrow", "I’ll send the report by evening",
        "Thanks for your help", "See you at the gym",
        "Call me when you’re free", "Family dinner this weekend?",
        "Project deadline is next week", "Good luck on your exam"
    ],
    "Label": [1,1,1,1,1,1,1,1,1,1,
              0,0,0,0,0,0,0,0,0,0,
              0,0,0,0]
}

df = pd.DataFrame(data)


vectorizer=CountVectorizer(stop_words="english")

X=vectorizer.fit_transform(df["Messages"])
y=df["Label"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
model=LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

predictions=model.predict(X_test)

print("Prediction:", predictions)
print("Actual:", y_test.values)

acc=accuracy_score(y_test, predictions)
prec=precision_score(y_test, predictions)
rec=recall_score(y_test, predictions)
f1=f1_score(y_test, predictions)
cm=confusion_matrix(y_test, predictions)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("Confusion Matrix:\n", cm)
