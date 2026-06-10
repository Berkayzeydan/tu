import pickle
import os
from datetime import datetime

import Evaluate_Clustering
import pairwise_f1
import unsupervised_pipeline_gpu as unsupervised_pipeline


with open("Ml_Expermintation/gt_clusters_300.pkl", "rb") as f:
    large_dict = pickle.load(f)

with open("Ml_Expermintation/splits_300.pkl", "rb") as f:
    large_dfs = pickle.load(f)



def shuffle_df_and_clusters(df, gt_clusters, random_state=None):
    """
    Shuffle the DataFrame and remap gt_clusters to match the new indices.
    Args:
        df: DataFrame to shuffle
        gt_clusters: List of clusters (each cluster is a list of indices)
        random_state: Optional random seed
    Returns:
        shuffled_df: The shuffled DataFrame (with reset index)
        remapped_clusters: The clusters with indices remapped to the new DataFrame
    """
    # Shuffle and keep original indices
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=False)
    # Build mapping: old index -> new index (row number in shuffled_df)
    old_to_new = {row['index']: new_idx for new_idx, row in shuffled_df.iterrows()}
    remapped_clusters = [[old_to_new[idx] for idx in cluster] for cluster in gt_clusters]
    # Drop the extra 'index' column for downstream compatibility
    shuffled_df = shuffled_df.drop(columns=['index']).reset_index(drop=True)
    return shuffled_df, remapped_clusters

def save_results_pickle(results: dict, name: str, folder: str = "results/unsupervised"):
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(folder, f"{name}_{timestamp}.pkl")

    with open(path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {path}")


def run_unsupervised_large_experiments(seed=42):
    tokenizer, model = unsupervised_pipeline.load_qwen_model()

    dict_results = {name: [] for name in large_dict.keys()}

    for i in range(3):
        n_seed = seed + i
        print(f"Running unsupervised experiment with seed {n_seed}...")

        for dataset_name, gt_clusters in large_dict.items():
            print(f"Processing dataset: {dataset_name}")

            strings_df = large_dfs[dataset_name]["0"]

            strings_df, remapped_clusters = shuffle_df_and_clusters(
                strings_df,
                gt_clusters,
                random_state=n_seed
            )

            pred_clusters, usage, lat = unsupervised_pipeline.unsupervised_pipeline(
                strings_df,
                tokenizer,
                model,
                seed=n_seed
            )

            f1, precision, recall = pairwise_f1.pairwise_f1_score(
                remapped_clusters,
                pred_clusters
            )

            print(f"{dataset_name} | seed={n_seed} | F1={f1:.4f}")

            dict_results[dataset_name].append({
                "seed": n_seed,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "token_usage": usage,
                "latency": lat
            })

    return dict_results


if __name__ == "__main__":
    base_seeds = [42, 55, 66]

    for base_seed in base_seeds:
        print("=" * 80)
        print(f"Starting unsupervised run with base seed {base_seed}")
        print("=" * 80)

        results = run_unsupervised_large_experiments(seed=base_seed)

        save_results_pickle(
            results,
            name=f"unsupervised_results_base_seed_{base_seed}",
            folder="results/unsupervised"
        )
