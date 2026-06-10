import numpy as np
import pandas as pd
import random
import asyncio
from itertools import combinations
import pandas as pd
import numpy as np



def get_supervision_pairs(strings, true_clusters, n_samples=5, mode="balanced_101", seed=None):
    """
    Generates supervision data and returns sampled supervision pairs
    plus the remaining unsampled pairs and all combination pairs.
    """
    rng = random.Random(seed) if seed is not None else random

    # 1. Map index to cluster ID
    idx_to_cluster = {}
    for cluster_id, cluster in enumerate(true_clusters):
        for idx in cluster:
            idx_to_cluster[idx] = cluster_id
            
    # 2. Separate all possible pairs into Must-Link and Cannot-Link
    must_link = []
    cannot_link = []
    for i, j in combinations(range(len(strings)), 2):
        if idx_to_cluster.get(i) == idx_to_cluster.get(j):
            must_link.append((i, j, 1))
        else:
            cannot_link.append((i, j, 0))

    selected_indices = []

    # 3. Selection Logic based on Mode
    if mode == "only_ones":
        selected_indices = rng.sample(must_link, min(n_samples, len(must_link)))
    elif mode == "only_zeros":
        selected_indices = rng.sample(cannot_link, min(n_samples, len(cannot_link)))
    elif mode == "balanced_101":
        # Use sample instead of choice to avoid duplicates
        ones = rng.sample(must_link, (n_samples + 1) // 2)
        zeros = rng.sample(cannot_link, n_samples // 2)
        # Interleave
        for k in range(n_samples):
            selected_indices.append(ones.pop(0) if k % 2 == 0 else zeros.pop(0))
    elif mode == "balanced_010":
        zeros = rng.sample(cannot_link, (n_samples + 1) // 2)
        ones = rng.sample(must_link, n_samples // 2)
        for k in range(n_samples):
            selected_indices.append(zeros.pop(0) if k % 2 == 0 else ones.pop(0))
    else:
        raise ValueError("Invalid mode.")

    selected_set = set(selected_indices)
    all_pairs = must_link + cannot_link
    remaining_indices = [pair for pair in all_pairs if pair not in selected_set]

    def build_pair_dataframe(pairs):
        pair_rows = []
        for i, j, label in pairs:
            # Keep both index-based and text-based supervision fields so downstream
            # components can consume the same dataframe without adapters

            pair_rows.append({
                'idx1': i,
                'idx2': j,
                'label': label
            })
        return pd.DataFrame(pair_rows)

    supervision_df = build_pair_dataframe(selected_indices)
    remaining_df = build_pair_dataframe(remaining_indices)
    all_pairs_df = build_pair_dataframe(all_pairs)
    return supervision_df, remaining_df, all_pairs_df



def initialize_datasets(df:pd.DataFrame,
                        supervision_pairs:pd.DataFrame,
                        remaining_pairs:pd.DataFrame,
                        all_combination_pairs:pd.DataFrame,
                    schema_model_name: str = "gpt-4o",
                    schema_reasoning_effort: str = "low",
                    schema_verbosity: str = "low",
                    seed: int = 42,
                    tag_model_name: str = "gpt-4o",
                    tag_reasoning_effort: str = "low",
                    batch_size: int = 20
                    ):
    """
    Initializes the dataset and pair features for Machine Learning Models.
    Returns supervision features, prediction features, all-pairs features,
    and usage statistics.
    """
    strings = df["0"].to_list()
    sampled_strings = random.Random(seed).sample(strings, 10)
    discovered_tags, schema_usage = Prompts.discover_schema_v2(strings=sampled_strings,
                                                model_name=schema_model_name,
                                                reasoning_effort=schema_reasoning_effort,
                                                verbosity=schema_verbosity,
                                                return_usage=True)
    print(f"Discovered Tags: {discovered_tags}")
    df, tag_usage = asyncio.run(
        Prompts.process_full_dataframe(
            df=df,
            tag_list=discovered_tags,
            batch_size=batch_size,
            model_name=tag_model_name,
            reasoning_effort=tag_reasoning_effort,
        )
    )
    df["compressed_topei"] = df["0"].apply(Topei.get_topei_rep)
    df["uncompressed_topei"] = df["0"].apply(Topei.get_uncompressed_topei)
    df["separators_of_symbol"] = df["0"].apply(Topei.separators_of_symbol)

    def build_feature_dataframe(pairs_df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []
        for _, pair in pairs_df.iterrows():
            idx1, idx2, label = pair.idx1, pair.idx2, pair.label

            comp_top1, comp_top2 = df.loc[idx1, "compressed_topei"], df.loc[idx2, "compressed_topei"]
            uncomp_top1, uncomp_top2 = df.loc[idx1, "uncompressed_topei"], df.loc[idx2, "uncompressed_topei"]
            sep_top1, sep_top2 = df.loc[idx1, "separators_of_symbol"], df.loc[idx2, "separators_of_symbol"]
            pat_seq1, pat_seq2 = df.loc[idx1, "tag_list"], df.loc[idx2, "tag_list"]

            feature_rows.append({
                "idx1": idx1,
                "idx2": idx2,
                "feat_compressed_ratio": Topei.feat_compressed_ratio(comp_top1, comp_top2),
                "feat_uncompressed_ratio": Topei.feat_uncompressed_ratio(uncomp_top1, uncomp_top2),
                "feat_symbol_sim": Topei.feat_symbol_sim(sep_top1, sep_top2),
                "feat_semantic_exact_match": Prompts.feat_semantic_exact_match(pat_seq1, pat_seq2),
                "feat_semantic_jaccard_sim": Prompts.feat_semantic_jaccard_sim(pat_seq1, pat_seq2),
                "feat_semantic_common_order_lcs_ratio": Prompts.feat_semantic_common_order_lcs_ratio(pat_seq1, pat_seq2),
                "label": label,
            })
        return pd.DataFrame(feature_rows)

    supervision_features_df = build_feature_dataframe(supervision_pairs)
    remaining_features_df = build_feature_dataframe(remaining_pairs)
    all_pairs_features_df = build_feature_dataframe(all_combination_pairs)
    
    usage_report = {
        "schema_usage": schema_usage,
        "tag_usage": tag_usage,
    }

    return supervision_features_df, remaining_features_df, all_pairs_features_df, usage_report



def initialize_datasets_v2(df:pd.DataFrame,
                    schema_model_name: str = "gpt-4o",
                    schema_reasoning_effort: str = "low",
                    schema_verbosity: str = "low",
                    seed: int = 42,
                    tag_model_name: str = "gpt-4o",
                    tag_reasoning_effort: str = "low",
                    batch_size: int = 20,
                    features_to_use: dict={}
                    ):
 
    """
    Applies selected features to a string column and returns a transformed dataset.

    `features_to_use` must be a dict in the form:
    {"new_col_name": callable}
    """
    result_df = df.copy()
    strings = df["0"].to_list()
    sampled_strings = random.Random(seed).sample(strings, 20)
    discovered_tags, schema_usage = Prompts.discover_schema_v2(strings=sampled_strings,
                                                 model_name=schema_model_name,
                                                reasoning_effort=schema_reasoning_effort,
                                                verbosity=schema_verbosity,
                                                return_usage=True)
    print(f"Discovered Tags: {discovered_tags}")
    result_df, tag_usage = asyncio.run(
        Prompts.process_full_dataframe(
            df=df,
            tag_list=discovered_tags,
            batch_size=batch_size,
            model_name=tag_model_name,
            reasoning_effort=tag_reasoning_effort,
        )
    )
    result_df["compressed_topei"] = result_df["0"].apply(Topei.get_topei_rep)
    result_df["uncompressed_topei"] = result_df["0"].apply(Topei.get_uncompressed_topei)
    result_df["separators_of_symbol"] = result_df["0"].apply(Topei.separators_of_symbol)
    result_df["semantic_ordered_tag_pairs"] = result_df["tag_list"].apply(Prompts.get_ordered_tag_pairs)
    
    for feature_name, feature_fn in features_to_use.items():
        result_df[feature_name] = result_df["0"].apply(feature_fn)

    return result_df

if __name__ == "__main__":
    pass