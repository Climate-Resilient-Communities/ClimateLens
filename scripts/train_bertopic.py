import traceback
from climatelens.utils.load_env import load_environment
data_dir, code_dir, JUPYTER = load_environment()
from climatelens.utils import process_datasets, create_directories

from climatelens.nlp_pipeline import topic_modeling as tm
parameters = tm.DATASET_PARAMS
bert_model = tm.bert_model
save_dataframe = tm.save_dataframe_inplace
save_model = tm.save_and_reload_model

from climatelens.nlp_pipeline.postprocessing import annotate_data, process_topic_merges, update_model
from climatelens.nlp_pipeline.embeddings import compute_embeddings
from climatelens.nlp_pipeline.dynamic_topic_modeling import run_dynamic_topic_modeling

def main():
    dfs, docs_dict, datasets = process_datasets(data_dir)

    model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir = create_directories(code_dir)
    dirs = (model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir)

    embeddings_dict, embedding_models = compute_embeddings(docs_dict)

    topic_models, topics_dict, probs_dict = {}, {}, {}
    topic_info_dict, core_topics_dict = {}, {}

    # Train models
    for name, docs in docs_dict.items():
        try:
            print("\n" + "=" * 60 + f"\n{name.upper()}:\n")
            params = parameters.get(name, parameters["reddit"])
            topic_model, topics, probs = bert_model(
                dataset_name=name,
                docs=docs,
                embeddings=embeddings_dict[name],
                embedding_model=embedding_models[name],
                params=params,
            )

        except ValueError as e:
            if "After pruning, no terms remain" not in str(e):
                raise  # re-raise unexpected ValueErrors

            # popular BERTopic / c-TF-IDF failure. using fallback params
            print(f"Using smaller reddit parameters for {name}\n")
            params = parameters.get(name, parameters["reddit_small"])
            topic_model, topics, probs = bert_model(
                dataset_name=name,
                docs=docs,
                embeddings=embeddings_dict[name],
                embedding_model=embedding_models[name],
                params=params,
            )

        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs

        if topic_models[name] is None:
            print(f"Skipping {name}: no valid topic model")
            continue

    # Post-process & annotate
    for name in dfs.keys():
        if topic_models.get(name) is None:
            print(f"Skipping post-processing for {name} since there's no topic model")
            continue
        topic_info_dict[name] = topic_models[name].get_topic_info()
        annotate_data(
            dfs, name, JUPYTER,
            topics_dict, probs_dict, topic_info_dict=topic_info_dict
        )
        process_topic_merges(dfs, topic_info_dict, name)

    # Update models and generate static visualizations
    for name in dfs.keys():
        try:
            params = parameters.get(name, parameters["reddit"])
            topic_models[name] = update_model(
                name=name,
                dfs=dfs,
                docs=docs_dict[name],
                topic_models=topic_models,
                docs_dict=docs_dict,
                dirs=dirs,
                core_topics_dict=core_topics_dict,
                topics_dict=topics_dict,
                probs_dict=probs_dict,
                nr_topics=params["nr_topics"],
            )
            save_dataframe(datasets[name], dfs[name])
            save_model(name, model_dir, topic_models)

        except ValueError as e:
            if "After pruning, no terms remain" not in str(e):
                # if it's unexpected, log and continue (don't kill entire pipeline)
                print(f"Unexpected ValueError when updating {name}: {e}")
                traceback.print_exc()
                continue

            print(f"Using smaller reddit parameters for {name}\n")
            try:
                params = parameters.get(name, parameters["reddit_small"])
                topic_models[name] = update_model(
                    name=name,
                    dfs=dfs,
                    docs=docs_dict[name],
                    topic_models=topic_models,
                    docs_dict=docs_dict,
                    dirs=dirs,
                    core_topics_dict=core_topics_dict,
                    topics_dict=topics_dict,
                    probs_dict=probs_dict,
                    nr_topics=params["nr_topics"],
                )
                save_dataframe(datasets[name], dfs[name])
                save_model(name, model_dir, topic_models)
            except Exception:
                print(f"Failed updating {name} even with smaller params; skipping.")
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"Unexpected error when updating model {name}: {e}")
            traceback.print_exc()
            # continue to next dataset instead of aborting the whole pipeline
            continue

    # Running dtm
    try:
        run_dynamic_topic_modeling(
            dfs=dfs,
            topic_models=topic_models,
            docs_dict=docs_dict,
            dtm_dir=dtm_dir
        )
    except Exception as e:
        print(f"DTM stage failed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Exception in pipeline:")
        traceback.print_exc()
        pass