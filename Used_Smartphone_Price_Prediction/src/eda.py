import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data_utils import load_project_data, find_target
from src.config import FIGURE_DIR

def main():
    df = load_project_data()
    target = find_target(df.columns)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(pd.to_numeric(df[target], errors="coerce").dropna(), kde=True, color="steelblue")
    plt.title(f"Distribution of {target}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_target_distribution.png", dpi=160)
    plt.close()

    # Numeric correlations
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "02_correlation_heatmap.png", dpi=160)
        plt.close()

    # Brand distribution / price by brand
    brand = next((c for c in df.columns if c.lower() in
                  {"brand_name", "brand", "manufacturer"}), None)
    if brand:
        top = df[brand].value_counts().head(15).index
        plt.figure(figsize=(11, 5))
        sns.countplot(data=df[df[brand].isin(top)], y=brand,
                      order=df[brand].value_counts().head(15).index, color="steelblue")
        plt.title("Most Frequent Smartphone Brands")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "03_brand_distribution.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df[brand].isin(top)], x=brand, y=target, color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{target} by Brand")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "04_price_by_brand.png", dpi=160)
        plt.close()

    # Numeric feature vs target scatter plots
    likely = ["days_used", "release_year", "ram", "internal_memory",
              "battery", "screen_size", "rear_camera_mp", "front_camera_mp"]
    for col in likely:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[target], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum():
                plt.figure(figsize=(8, 5))
                sns.scatterplot(x=x[valid], y=y[valid], alpha=0.45, color="steelblue")
                plt.xlabel(col)
                plt.ylabel(target)
                plt.title(f"{target} vs {col}")
                plt.tight_layout()
                plt.savefig(FIGURE_DIR / f"05_{col}_vs_target.png", dpi=160)
                plt.close()

    print(f"EDA charts saved in: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
