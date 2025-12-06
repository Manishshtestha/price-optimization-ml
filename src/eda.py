def advanced_eda(df):
    return {
        "summary": df.describe(),
        "missing_ratio": df.isnull().mean(),
        "corr": df.corr(numeric_only=True)
    }


def plot_numeric_distributions(df, bins=30):
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].hist(bins=bins, figsize=(12, 8))
    plt.tight_layout()
    plt.show()
