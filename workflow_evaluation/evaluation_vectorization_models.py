import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import ast

def test():
    df1 = pd.read_pickle("../vectorization_process_files/list_files/biz_document_vectors_tfdoc.pkl")

    df2 = pd.read_pickle("../workflow_files/list_files/document_vectors.pkl")

    df_daily = pd.merge(df1, df2, on="added_date", how="inner")

    df_stock = pd.read_csv("../stock_price_files/HNB_NEW.csv", parse_dates=['Date'])

    df_stock.fillna(0, inplace=True)
    df_stock['price'] = df_stock['price'].replace(0, np.nan)
    df_stock['price'] = df_stock['price'].fillna(method='ffill').fillna(method='bfill')

    df_stock['return'] = df_stock['price'].diff()
    df_stock.dropna(inplace=True)

    Q1 = df_stock['price'].quantile(0.25)
    Q3 = df_stock['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df_stock[(df_stock['price'] >= lower_bound) & (df_stock['price'] <= upper_bound)]

    # Ensure `df_daily` has a date column named `added_date`
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    df_daily['added_date'] = pd.to_datetime(df_daily['added_date'])
    df_daily = df_daily.merge(filtered_df, left_on='added_date', right_on='Date', how='left')

    # Drop redundant date column
    df_daily = df_daily.drop(columns=["Date", "Open", "High", "Low", "Vol."], axis=1)

    # Ensure no missing values
    df_daily.dropna(inplace=True)

    # df_daily.to_pickle("workflow_files/list_files/price_attached_vectors.pkl")
    df_daily.to_pickle("evaluation_vectorization_models/price_attached_all_vectors.pkl")



def plot():
    df = pd.read_pickle("../evaluation_vectorization_models/price_attached_all_vectors.pkl")
    # --- Utility: Convert stringified vectors to NumPy arrays if needed ---
    def convert_to_array(series):
        if isinstance(series.iloc[0], str):
            return series.apply(lambda x: np.array(ast.literal_eval(x)))
        return series

    # Load and process DataFrame
    # df = pd.read_pickle("your_file.pkl")  # Already loaded as df

    # Convert all vectors
    df["tfidf_vectors"] = convert_to_array(df["tfidf_vectors"])
    df["doc2vec_vectors"] = convert_to_array(df["doc2vec_vectors"])
    df["sbert_vectors"] = convert_to_array(df["sbert_vectors"])
    df["fingpt_vectors"] = convert_to_array(df["fingpt_vectors"])

    # Convert to stacked matrices
    tfidf_matrix = np.stack(df["tfidf_vectors"])
    doc2vec_matrix = np.stack(df["doc2vec_vectors"])
    sbert_matrix = np.stack(df["sbert_vectors"])
    fingpt_matrix = np.stack(df["fingpt_vectors"])

    # --- Labeling: Bin 'return' into categories ---
    df["return_label"] = pd.cut(df["return"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["Down", "Neutral", "Up"])
    labels = df["return_label"].values

    # --- Choose dimensionality reduction method ---
    use_umap = False  # Set to True to use UMAP

    def reduce_and_plot(vectors, title, labels):
        if use_umap:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)

        vectors_2d = reducer.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(vectors_2d[idx, 0], vectors_2d[idx, 1], label=label, alpha=0.6)
        plt.title(f"{title} Embeddings - {'UMAP' if use_umap else 't-SNE'}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Run Visualizations ---
    reduce_and_plot(tfidf_matrix, "TF-IDF", labels)
    reduce_and_plot(doc2vec_matrix, "Doc2Vec", labels)
    reduce_and_plot(sbert_matrix, "SBERT", labels)
    reduce_and_plot(fingpt_matrix, "FinGPT", labels)


def plot2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import ast

    df = pd.read_pickle("../evaluation_vectorization_models/price_attached_all_vectors.pkl")
    # --- Convert stringified vectors (if needed) ---
    def convert_to_array(series):
        if isinstance(series.iloc[0], str):
            return series.apply(lambda x: np.array(ast.literal_eval(x)))
        return series

    # === Step 1: Preprocess ===
    # Filter strong Up/Down only
    df_extreme = df[(df['return'] > 1.0) | (df['return'] < -1.0)].copy()
    df_extreme['return_label'] = df_extreme['return'].apply(lambda x: 'Up' if x > 1.0 else 'Down')

    # Convert embeddings
    for col in ['tfidf_vectors', 'doc2vec_vectors', 'sbert_vectors', 'fingpt_vectors']:
        df_extreme[col] = convert_to_array(df_extreme[col])

    # Stack embeddings
    tfidf_matrix = np.stack(df_extreme['tfidf_vectors'])
    doc2vec_matrix = np.stack(df_extreme['doc2vec_vectors'])
    sbert_matrix = np.stack(df_extreme['sbert_vectors'])
    fingpt_matrix = np.stack(df_extreme['fingpt_vectors'])

    labels = df_extreme['return_label'].values

    # === Step 2: UMAP + Plot ===
    def reduce_and_plot(vectors, title, labels):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        vectors_2d = reducer.fit_transform(vectors)

        plt.figure(figsize=(7, 6))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(vectors_2d[idx, 0], vectors_2d[idx, 1], label=label, alpha=0.7)
        plt.title(f"{title} Embeddings (UMAP)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Step 3: Run visualizations ===
    reduce_and_plot(tfidf_matrix, "TF-IDF", labels)
    reduce_and_plot(doc2vec_matrix, "Doc2Vec", labels)
    reduce_and_plot(sbert_matrix, "SBERT", labels)
    reduce_and_plot(fingpt_matrix, "FinGPT", labels)


test()
plot2()

