import pandas as pd
from scipy.stats import ttest_rel  # For paired t-test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# 1. Prepare Your Data and Cross-Validation Setup:
# ... (Load your data X and y) ...
df_train = pd.read_pickle("../vectorization_process_files/list_files/train.pkl")
X = df_train["tfidf_vectors"]
y = df_train["price"]

# 1. Prepare Data and Cross-Validation:
# ... (Load your data X and y) ...

k = 5  # Example: 5-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)  # *Crucial*: Consistent splits!

# 2. Define Pipelines with Different TF-IDF Configurations:

# Configuration 1:
tfidf1 = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,
    min_df=5,
    sublinear_tf=True,
    norm='l2',
    ngram_range=(1, 2),
    max_features=200
)
pipeline1 = Pipeline([('tfidf', tfidf1), ('clf', LogisticRegression(random_state=42))])

# Configuration 2:
tfidf2 = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=5,
    sublinear_tf=True,
    norm='l2',
    ngram_range=(1, 2),
    max_features=400
)
pipeline2 = Pipeline([('tfidf', tfidf2), ('clf', LogisticRegression(random_state=42))])

# 3. Cross-Validation and Scores:

scores1 = cross_val_score(pipeline1, X, y, cv=kf, scoring='accuracy')  # Same folds!
scores2 = cross_val_score(pipeline2, X, y, cv=kf, scoring='accuracy')  # Same folds!

# 4. Paired t-test:

t_statistic, p_value = ttest_rel(scores1, scores2)

# 5. Interpret:

alpha = 0.05

print("Configuration 1 Mean:", scores1.mean())
print("Configuration 2 Mean:", scores2.mean())
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < alpha:
    print("The difference is statistically significant.")
    if scores2.mean() > scores1.mean():
        print("Configuration 2 is better.")
    else:
        print("Configuration 1 is better.")
else:
    print("No statistically significant difference.")

# 6. (Optional) Print all scores
print("Scores for Configuration 1:", scores1)
print("Scores for Configuration 2:", scores2)
