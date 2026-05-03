"""
Usage::

topic_models[name] = update_model(
    name=name,
    dfs=dfs,
    topic_models=topic_models,
    docs_dict=docs_dict,
    core_topics_dict=core_topics_dict,
    topics_dict=topics_dict,
    probs_dict=probs_dict,
    nr_topics=params["nr_topics"],
)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from climatelens.utils import create_directories


def annotate_data(dfs, name, topics_dict, probs_dict, topic_info_dict):
    dfs[name]["topic"] = topics_dict[name]
    dfs[name]["topic_proba"] = probs_dict[name]

    print(f"\nNumber of topics (including outlier): {len(topic_info_dict[name])}")

def clean_dataframe_columns(df, name):
    # Remove large columns and existing merge columns to avoid duplicates
    cols_to_remove = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]

    # large columns
    large_cols = [
        'Representation', 'Representative_Docs',
        'Representation_core', 'Representative_Docs_core',
        'Name', 'Name_core'
    ]

    cols_to_remove.extend([col for col in large_cols if col in df.columns])

    topic_cols = [col for col in df.columns if col.startswith('Topic_')]
    cols_to_remove.extend(topic_cols)

    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} duplicate/artifact columns from {name}")

        df = df.drop(columns=cols_to_remove, errors='ignore')

    return df


def process_topic_merges(dfs, topic_info_dict, name, topic_col="topic", repr_docs_col="Representative_Docs"):
    """
    Create representative flag WITHOUT merging large columns into main DF.
    """
    # Clean up any existing artifact columns first
    dfs[name] = clean_dataframe_columns(dfs[name], name)

    # Make sure we have the topic column
    if topic_col not in dfs[name].columns:
        print(f"Warning: {topic_col} not found in {name}")
        return dfs[name]

    # Create representative flag without merging large columns
    def is_representative(row):
        if not isinstance(row.get(topic_col), (int, float)):
            return 0
        # Find the topic row in topic_info_dict
        topic_row = topic_info_dict[name][topic_info_dict[name]["Topic"] == row[topic_col]]
        if topic_row.empty:
            return 0
        repr_docs = topic_row.iloc[0].get(repr_docs_col, [])
        if isinstance(repr_docs, list) and row.get("cleaned_text") in repr_docs:
            return 1
        return 0

    is_repr_col = f"is_representative{'_core' if 'core' in topic_col else ''}"
    dfs[name][is_repr_col] = dfs[name].apply(is_representative, axis=1)

    return dfs[name]


def process_core_topics(dfs, name, core_topics_df, topics_dict, probs_dict):
    """
    Add core topic info WITHOUT merging large columns into main DF.
    """
    # Clean up existing columns first
    dfs[name] = clean_dataframe_columns(dfs[name], name)

    # Add core topic IDs and probabilities to main DF (small numeric columns)
    dfs[name]["core_topic"] = topics_dict[name]
    dfs[name]["core_topic_proba"] = probs_dict[name]

    # Create a flag for representative docs without storing the lists
    def is_representative_core(row):
        if not isinstance(row.get("core_topic"), (int, float)):
            return 0
        # Find representative docs for this topic
        topic_row = core_topics_df[core_topics_df["Topic"] == row["core_topic"]]
        if topic_row.empty:
            return 0
        repr_docs = topic_row.iloc[0].get("Representative_Docs", [])
        if isinstance(repr_docs, list) and row.get("cleaned_text") in repr_docs:
            return 1
        return 0

    dfs[name]["is_representative_core"] = dfs[name].apply(is_representative_core, axis=1)

    # Return the core_topics_df as-is (for any future use, but NOT merged)
    return core_topics_df


def finalize_dataframe(df):
    """
    Remove all duplicate and large columns from the main dataframe.
    No JSON files created - just clean the dataframe.
    """
    # Remove all _x, _y suffix columns
    cols_to_remove = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]

    # Remove large list columns
    large_cols = [
        'Representation', 'Representative_Docs',
        'Representation_core', 'Representative_Docs_core',
        'Name', 'Name_core', 'Topic'
    ]
    cols_to_remove.extend([col for col in large_cols if col in df.columns])

    # Remove any columns that start with Topic_ (duplicate)
    topic_cols = [col for col in df.columns if col.startswith('Topic_')]
    cols_to_remove.extend(topic_cols)

    # Remove duplicates
    cols_to_remove = list(set(cols_to_remove))

    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} duplicate/large columns")
        df_clean = df.drop(columns=cols_to_remove, errors='ignore')
    else:
        df_clean = df.copy()

    print(f"Final columns kept: {', '.join(df_clean.columns)}")

    return df_clean


def save_dataframe_inplace(path, df):
    """
    Save dataframe after removing duplicate and large columns.
    """
    # Clean the dataframe first
    df_clean = finalize_dataframe(df)

    try:
        # Get original size for comparison
        original_size = path.stat().st_size / (1024 * 1024) if path.exists() else 0

        df_clean.to_csv(path, index=False)
        new_size = path.stat().st_size / (1024 * 1024)

        print(f"Saved to {path}")
        print(f"  - Original size: {original_size:.2f} MB" if original_size else "  - New file created")
        print(f"  - New size: {new_size:.2f} MB")
        print(f"  - Reduction: {(original_size - new_size):.2f} MB ({(1 - new_size/original_size)*100:.1f}% smaller)"
              if original_size else "")

        return True
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        return False


def update_model(
    name,
    dfs,
    topic_models,
    docs_dict,
    core_topics_dict,
    topics_dict,
    probs_dict,
    nr_topics=30,
):
    # Create only the directories you need for visualizations
    code_dir = Path(os.getenv("CODE_DIR", "."))
    paths = create_directories(
        code_dir / "outputs",
        [
            "visualizations/IDM",
            "visualizations/hierarchies",
            "visualizations/barcharts"
        ],
        use_timestamp=True
    )

    IDM_dir = paths["visualizations/IDM"]
    hierarchy_dir = paths["visualizations/hierarchies"]
    barchart_dir = paths["visualizations/barcharts"]

    topic_model = topic_models[name]

    # Clean dataframe before any merging operations
    dfs[name] = clean_dataframe_columns(dfs[name], name)

    topic_model_clustered = topic_model.reduce_topics(docs_dict[name], nr_topics=nr_topics)
    topic_model_clustered.update_topics(docs_dict[name], n_gram_range=(3, 5))

    core_topics = topic_model_clustered.get_topic_info()

    # Process core topics without merging large columns into main DF
    core_topics_metadata = process_core_topics(dfs, name, core_topics, topics_dict, probs_dict)
    core_topics_dict[name] = core_topics_metadata

    # Generate visualizations
    figure_hierarchy = topic_model_clustered.visualize_hierarchy()
    figure_topics = topic_model_clustered.visualize_topics()
    figure_barchart = topic_model_clustered.visualize_barchart(
        top_n_topics=len(core_topics), n_words=10
    )

    # Resize figures
    WIDTH = 1800
    HEIGHT = 1000

    figure_hierarchy.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Hierarchy")
    figure_topics.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Map")
    figure_barchart.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Barchart")

    figure_hierarchy.write_html(os.path.join(hierarchy_dir, f"{name}HRC.html"))
    figure_topics.write_html(os.path.join(IDM_dir, f"{name}IDM.html"))
    figure_barchart.write_html(os.path.join(barchart_dir, f"{name}BRC.html"))

    return topic_model_clustered
