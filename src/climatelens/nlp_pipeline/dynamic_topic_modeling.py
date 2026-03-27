# Original Author: Ardavan Shahrabi

def prepare_timestamps(dfs, name):
    """
    Extract and validate timestamps from a dataset.

    Handles different timestamp formats:
    - Twitter: 'created_at' column with datetime strings
    - Reddit: 'created_utc' column with Unix timestamps

    Args:
        dfs: Dictionary of dataframes
        name: Dataset name (key in dfs)

    Returns:
        List of datetime objects corresponding to each document, or None if no timestamps found
    """
    df = dfs[name]
    timestamps = None

    # Possible timestamp column names (prioritized)
    timestamp_cols = ['created_utc', 'created_at', 'timestamp', 'date', 'datetime']

    found_col = None
    for col in timestamp_cols:
        if col in df.columns:
            found_col = col
            break

    if not found_col:
        print(f"No timestamp column found for {name}. Checked: {timestamp_cols}")
        print(f"Available columns: {list(df.columns)}")
        print(f"Skipping Dynamic Topic Modeling for {name}.")
        return None

    print(f" Found timestamp column '{found_col}' for {name}")

    try:
        if found_col == 'created_utc':
            # Reddit uses Unix timestamps (seconds since epoch)
            timestamps = pd.to_datetime(df[found_col], unit='s', errors='coerce')
        elif found_col == 'created_at':
            # Twitter uses datetime strings
            timestamps = pd.to_datetime(df[found_col], errors='coerce')
        else:
            # Try automatic parsing for other column names
            timestamps = pd.to_datetime(df[found_col], errors='coerce')

        # Convert to Python datetime objects (list)
        timestamps = timestamps.tolist()

        # Validate timestamps
        valid_timestamps = [t for t in timestamps if pd.notna(t)]
        invalid_count = len(timestamps) - len(valid_timestamps)

        if invalid_count > 0:
            print(f"{invalid_count}/{len(timestamps)} timestamps could not be parsed")

        if len(valid_timestamps) == 0:
            print(f"No valid timestamps found for {name}. Skipping DTM.")
            return None

        # Print time range info
        min_date = min(valid_timestamps)
        max_date = max(valid_timestamps)
        time_span_days = (max_date - min_date).days

        print(f"Time range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"Time span: {time_span_days} days (~{time_span_days/365:.1f} years)")

        # Check if time span is meaningful for DTM
        if time_span_days < 7:
            print(f"Time span less than 1 week. DTM may not be meaningful.")

        return timestamps

    except Exception as e:
        print(f"Error parsing timestamps for {name}: {e}")
        traceback.print_exc()
        return None

def calculate_optimal_bins(timestamps, min_bins=10, max_bins=50):
    """
    Calculate optimal number of temporal bins based on data time span.

    Heuristic: ~1 bin per month, bounded by min/max limits.

    Args:
        timestamps: List of datetime objects
        min_bins: Minimum number of bins
        max_bins: Maximum number of bins

    Returns:
        Integer number of bins
    """
    valid_timestamps = [t for t in timestamps if pd.notna(t)]

    if len(valid_timestamps) < 2:
        return min_bins

    min_date = min(valid_timestamps)
    max_date = max(valid_timestamps)
    time_span_days = (max_date - min_date).days

    # Roughly 1 bin per month (30 days)
    suggested_bins = max(1, time_span_days // 30)

    # Clamp to min/max bounds
    optimal_bins = max(min_bins, min(max_bins, suggested_bins))

    return optimal_bins

def perform_dynamic_topic_modeling(topic_model, docs, timestamps, name, nr_bins=None, top_n_topics=10):
    """
    Perform Dynamic Topic Modeling analysis using BERTopic's topics_over_time.

    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        timestamps: List of datetime objects (same order as docs)
        name: Dataset name for logging
        nr_bins: Number of temporal bins (auto-calculated if None)
        top_n_topics: Number of topics to include in visualization

    Returns:
        Tuple of (topics_over_time DataFrame, Plotly figure) or (None, None) on failure
    """
    if timestamps is None:
        return None, None

    # Calculate optimal bins if not specified
    if nr_bins is None:
        nr_bins = calculate_optimal_bins(timestamps)

    print(f"\n Performing Dynamic Topic Modeling for {name}...")
    print(f"Using {nr_bins} temporal bins, visualizing top {top_n_topics} topics")

    start_time = time.time()

    try:
        # BERTopic's topics_over_time handles binning and aggregation
        topics_over_time = topic_model.topics_over_time(
            docs=docs,
            timestamps=timestamps,
            nr_bins=nr_bins,
            datetime_format=None,  # Auto-detect format
            evolution_tuning=True,  # Fine-tune topic representations per time bin
            global_tuning=True      # Use global topic representations as reference
        )

        # Generate interactive visualization
        fig = topic_model.visualize_topics_over_time(
            topics_over_time=topics_over_time,
            top_n_topics=top_n_topics,
            normalize_frequency=False,
            title=f"Topic Evolution Over Time - {name}"
        )

        elapsed_time = time.time() - start_time
        print(f"DTM completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(topics_over_time)} temporal data points")

        return topics_over_time, fig

    except Exception as e:
        print(f"Error during DTM for {name}: {e}")
        traceback.print_exc()
        return None, None

def save_dtm_outputs(topics_over_time, fig, name, dtm_dir):
    """
    Save DTM outputs: CSV data and HTML visualization.

    Args:
        topics_over_time: DataFrame from topics_over_time
        fig: Plotly figure from visualize_topics_over_time
        name: Dataset name for file naming
        dtm_dir: Output directory path
    """
    if topics_over_time is None or fig is None:
        print(f"No DTM outputs to save for {name}")
        return

    try:
        # Save CSV data for further analysis
        csv_path = Path(dtm_dir) / f"{name}_topics_over_time.csv"
        topics_over_time.to_csv(csv_path, index=False)
        print(f"Saved DTM data: {csv_path}")

        # Save interactive HTML visualization
        html_path = Path(dtm_dir) / f"{name}_topics_over_time.html"
        fig.write_html(str(html_path))
        print(f"Saved DTM visualization: {html_path}")

    except Exception as e:
        print(f"Error saving DTM outputs for {name}: {e}")
        traceback.print_exc()

def run_dynamic_topic_modeling(dfs, topic_models, docs_dict, dtm_dir):
    print("\n" + "=" * 60)
    print("Starting Dynamic Topic Modeling:\n")
    print("=" * 60)

    for name in dfs.keys():
        print(f"\n{'─' * 40}")
        print(f"Processing {name} for DTM...")

        # Prepare timestamps
        timestamps = prepare_timestamps(dfs, name)

        if timestamps is None:
            print(f"Skipping DTM for {name} (no valid timestamps)")
            continue

        try:
            #Use existing trained model
            model = topic_models.get(name)
            if model is None:
                print(f"Skipping DTM for {name} (no trained model)")
                continue

            topics_over_time, fig = perform_dynamic_topic_modeling(
                topic_model=model,
                docs=docs_dict[name],
                timestamps=timestamps,
                name=name,
                nr_bins=None,
                top_n_topics=10
            )

            # Save outputs
            if topics_over_time is not None and fig is not None:
                save_dtm_outputs(topics_over_time, fig, name, dtm_dir)

        except Exception as e:
            print(f"DTM failed for {name}: {e}")
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("Dynamic Topic Modeling complete.")
    print("=" * 60)