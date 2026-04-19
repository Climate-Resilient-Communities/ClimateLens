import os


def annotate_data(dfs, name, JUPYTER, topics_dict, probs_dict, topic_info_dict):
    dfs[name]["topic"] = topics_dict[name]
    dfs[name]["topic_proba"] = probs_dict[name]

    if JUPYTER:
        from IPython.display import display
        print("Processed data (sample):\n")
        display(dfs[name].sample(n=min(3, len(dfs[name]))))

        print(f"\nNumber of topics (including outlier): {len(topic_info_dict[name])}\n")
        display(topic_info_dict[name].sample(n=min(4, len(topic_info_dict[name]))))

def process_topic_merges(dfs, topic_info_dict, name, topic_col="topic", repr_docs_col="Representative_Docs"):
    # Drop any existing merge columns to avoid duplicates
    cols_to_drop = [c for c in dfs[name].columns if c.endswith('_x') or c.endswith('_y') or c in ['Name', 'Representation', 'Representative_Docs']]
    dfs[name] = dfs[name].drop(columns=cols_to_drop, errors='ignore')
    df = dfs[name].merge(
        topic_info_dict[name][["Topic", "Name", "Representation", repr_docs_col]],
        left_on=topic_col,
        right_on="Topic",
        how="left",
    )
    if "Topic" in df.columns:
      del df["Topic"]

    is_repr_col = f"is_representative{'_core' if 'core' in topic_col else ''}"
    df[is_repr_col] = df.apply(
        lambda row: 1
        if isinstance(row.get(repr_docs_col), list) and row.get("cleaned_text") in row.get(repr_docs_col)
        else 0,
        axis=1,
    )
    return df

def process_core_topics(dfs, name, core_topics, topics_dict, probs_dict):
    dfs[name]["core_topic"] = topics_dict[name]
    dfs[name]["core_topic_proba"] = probs_dict[name]

    core_topics = core_topics.rename(
        columns={
            "Name": "Name_core",
            "Representation": "Representation_core",
            "Representative_Docs": "Representative_Docs_core",
        }
    )

    dfs[name] = dfs[name].merge(
        core_topics[["Topic", "Name_core", "Representation_core", "Representative_Docs_core"]],
        left_on="core_topic",
        right_on="Topic",
        how="left",
    )

    dfs[name]["is_representative_core"] = dfs[name].apply(
        lambda row: 1
        if isinstance(row.get("Representative_Docs_core"), list)
        and row.get("cleaned_text") in row.get("Representative_Docs_core")
        else 0,
        axis=1,
    )

    return core_topics

def update_model(name, dfs, docs, topic_models, docs_dict, dirs, core_topics_dict, topics_dict, probs_dict, nr_topics=30):
    model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir = dirs  # Updated to include dtm_dir ????????????
    topic_model = topic_models[name]

    topic_model_clustered = topic_model.reduce_topics(docs_dict[name], nr_topics=nr_topics)
    topic_model_clustered.update_topics(docs_dict[name], n_gram_range=(3, 5))

    core_topics = topic_model_clustered.get_topic_info()
    core_topics = process_core_topics(dfs, name, core_topics, topics_dict, probs_dict)
    core_topics_dict[name] = core_topics

    figure_hierarchy = topic_model_clustered.visualize_hierarchy()
    figure_topics = topic_model_clustered.visualize_topics()
    figure_barchart = topic_model_clustered.visualize_barchart(top_n_topics=len(core_topics), n_words=10)

    # resizing figures to be larger
    WIDTH = 1800
    HEIGHT = 1000

    figure_hierarchy.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Hierarchy")
    figure_topics.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Map")
    figure_barchart.update_layout(width=WIDTH, height=HEIGHT, title=f"{name} Topic Barchart")

    figure_hierarchy.write_html(os.path.join(hierarchy_dir, f"{name}HRC.html"))
    figure_topics.write_html(os.path.join(IDM_dir, f"{name}IDM.html"))
    figure_barchart.write_html(os.path.join(barchart_dir, f"{name}BRC.html"))

    return topic_model_clustered
