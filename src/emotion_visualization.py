"""
Emotion Visualization Pipeline for Climate Anxiety Analysis

This module generates emotion-based visualizations from social media data:
- Word clouds for each emotion (7 emotions × 2 datasets = 14 files + 2 combined)
- Time-series visualizations showing emotion trends over time (4 HTML files)
- Summary report (1 HTML file)

The pipeline processes Twitter and Reddit datasets that have been pre-processed
with cleaned text and timestamps.

Usage:
    python emotion_visualizations.py

Environment Variables:
    DATA_DIR: Directory containing input CSV files (default: ./data)
    OUTPUT_DIR: Base directory for outputs (default: ./visualizations)

Author: Ardavan Shahrabi
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Emotion model configuration
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Visualization settings
EMOTION_COLORMAP = {
    'anger': 'Reds',
    'disgust': 'Greens',
    'fear': 'Purples',
    'joy': 'YlOrRd',
    'neutral': 'Greys',
    'sadness': 'Blues',
    'surprise': 'Oranges'
}

EMOTION_LINE_COLORS = {
    'anger': '#d62728',      # red
    'disgust': '#2ca02c',    # green
    'fear': '#9467bd',       # purple
    'joy': '#ff7f0e',        # orange
    'neutral': '#7f7f7f',    # gray
    'sadness': '#1f77b4',    # blue
    'surprise': '#e377c2'    # pink
}

# Required columns for each dataset type
REQUIRED_COLUMNS = {
    'twitter': ['created_at', 'text'],
    'reddit': ['body', 'created_utc']
}


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def load_environment() -> Dict[str, Path]:
    """
    Load environment configuration and determine data/output directories.
    
    Works in multiple environments:
    - Local development (uses DATA_DIR and OUTPUT_DIR env vars)
    - Google Colab (detects and uses Drive paths)
    - Azure/Cloud (uses environment variables)
    
    Returns:
        Dictionary with 'data_dir' and 'output_dir' paths
    """
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    # Check for environment variables first
    data_dir = os.environ.get('DATA_DIR')
    output_dir = os.environ.get('OUTPUT_DIR')
    
    if data_dir and output_dir:
        return {
            'data_dir': Path(data_dir),
            'output_dir': Path(output_dir)
        }
    
    # Check if running in Colab
    try:
        import google.colab
        # Running in Colab - use Drive paths
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        
        return {
            'data_dir': Path('/content/drive/MyDrive/ClimateLens/data'),
            'output_dir': Path('/content/drive/MyDrive/ClimateLens/visualizations')
        }
    except ImportError:
        pass
    
    # Default local paths
    return {
        'data_dir': Path('./data'),
        'output_dir': Path('./visualizations')
    }


def create_directories(base_output_dir: Path) -> Dict[str, Path]:
    """
    Create all necessary output directories.
    
    Args:
        base_output_dir: Base path for all outputs
        
    Returns:
        Dictionary mapping directory names to their paths
    """
    directories = {
        'emotions': base_output_dir / 'emotions',
        'wordclouds': base_output_dir / 'emotions' / 'wordclouds',
        'timeseries': base_output_dir / 'emotions' / 'timeseries',
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        
    return directories


# =============================================================================
# DATA LOADING
# =============================================================================

def find_csv_files(data_dir: Path) -> Dict[str, Path]:
    """
    Automatically find Twitter and Reddit CSV files in data directory.
    
    Args:
        data_dir: Directory to search for CSV files
        
    Returns:
        Dictionary with 'twitter' and 'reddit' file paths
        
    Raises:
        FileNotFoundError: If required files cannot be found
    """
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    found_files = {}
    missing_files = []
    
    # Find Twitter file
    twitter_file = next(
        (f for f in csv_files if 'twitter' in f.name.lower()),
        None
    )
    if twitter_file:
        found_files['twitter'] = twitter_file
    else:
        missing_files.append('Twitter (filename should contain "twitter")')
    
    # Find Reddit file
    reddit_file = next(
        (f for f in csv_files 
         if 'reddit' in f.name.lower() 
         or 'anticonsumption' in f.name.lower()
         or 'comments' in f.name.lower()
         or 'submissions' in f.name.lower()),
        None
    )
    if reddit_file:
        found_files['reddit'] = reddit_file
    else:
        missing_files.append('Reddit (filename should contain "reddit", "anticonsumption", "comments", or "submissions")')
    
    if missing_files:
        available = [f.name for f in csv_files]
        raise FileNotFoundError(
            f"Could not find required files:\n"
            f"  Missing: {missing_files}\n"
            f"  Available files: {available}"
        )
    
    return found_files


def validate_dataframe(df: pd.DataFrame, dataset_type: str, filename: str) -> List[str]:
    """
    Validate that a dataframe has required columns.
    
    Args:
        df: DataFrame to validate
        dataset_type: Either 'twitter' or 'reddit'
        filename: Name of the source file (for error messages)
        
    Returns:
        List of missing columns (empty if all present)
    """
    required = REQUIRED_COLUMNS.get(dataset_type, [])
    missing = [col for col in required if col not in df.columns]
    return missing


def load_datasets(file_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load and validate Twitter and Reddit datasets.
    
    Args:
        file_paths: Dictionary with 'twitter' and 'reddit' file paths
        
    Returns:
        Dictionary with 'Twitter' and 'Reddit' DataFrames
        
    Raises:
        KeyError: If required columns are missing
    """
    datasets = {}
    
    for dataset_type, filepath in file_paths.items():
        print(f"  Loading {dataset_type.capitalize()}: {filepath.name}")
        
        df = pd.read_csv(filepath)
        
        # Validate columns
        missing = validate_dataframe(df, dataset_type, filepath.name)
        if missing:
            raise KeyError(
                f"{dataset_type.capitalize()} dataset missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Use capitalized keys for consistency
        key = 'Twitter' if dataset_type == 'twitter' else 'Reddit'
        datasets[key] = df
        
        print(f"    Loaded {len(df):,} rows")
    
    return datasets


# =============================================================================
# EMOTION DETECTION
# =============================================================================

def load_emotion_model():
    """
    Load the HuggingFace emotion detection model.
    
    Returns:
        Tuple of (pipeline, device_name)
    """
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = 'GPU' if device == 0 else 'CPU'
    
    print(f"  Loading model: {EMOTION_MODEL_NAME}")
    print(f"  Using device: {device_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
    EMOTION_MODEL_NAME,
    use_safetensors=True  
)
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    emotion_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None  # Return all emotion scores
    )
    
    return emotion_classifier, device_name


def detect_emotions_batch(
    texts: List[str],
    emotion_classifier,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Detect emotions for a list of texts with batch processing.
    
    The model returns scores for all 7 emotions for each text.
    We extract the top emotion and also store individual scores.
    
    Args:
        texts: List of text strings to analyze
        emotion_classifier: HuggingFace pipeline for emotion detection
        batch_size: Number of texts to process at once
        
    Returns:
        DataFrame with columns:
        - emotion_label: Top predicted emotion
        - emotion_confidence: Confidence score for top emotion
        - anger, disgust, fear, joy, neutral, sadness, surprise: Individual scores
        
    Note:
        The pipeline returns a list of lists when top_k=None, where each inner
        list contains dicts with 'label' and 'score' for all emotions.
    """
    results = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        
        # Truncate long texts (model limit: 512 tokens)
        batch = [text[:512] if isinstance(text, str) else "" for text in batch]
        
        # Get predictions - returns list of lists when top_k=None
        predictions = emotion_classifier(batch)
        
        # Extract results for each text
        for pred in predictions:
            # pred is a list of dicts with 'label' and 'score'
            top_emotion = max(pred, key=lambda x: x['score'])
            
            # Create dict with all emotion scores
            emotion_scores = {item['label']: item['score'] for item in pred}
            
            results.append({
                'emotion_label': top_emotion['label'],
                'emotion_confidence': top_emotion['score'],
                **emotion_scores
            })
        
        # Progress indicator (every 10 batches)
        processed = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0:
            print(f"    Processed {processed:,}/{total:,} texts...")
    
    return pd.DataFrame(results)


def add_emotions_to_datasets(
    datasets: Dict[str, pd.DataFrame],
    emotion_classifier
) -> Dict[str, pd.DataFrame]:
    """
    Run emotion detection on all datasets and add results as new columns.
    Args:
        datasets: Dictionary of DataFrames to process
        emotion_classifier: HuggingFace pipeline
    Returns:
        Updated dictionary with emotion columns added to each DataFrame
    """
    updated_datasets = {}
    for name, df in datasets.items():
        print(f"\n  Processing {name} ({len(df):,} rows)...")
        # Get cleaned texts
        text_cols=('body', 'text')
        text_col = next((c for c in text_cols if c in df.columns), None)
        texts = df[text_col].fillna('').tolist()
        # Detect emotions
        emotion_results = detect_emotions_batch(texts, emotion_classifier)
        # Drop any existing emotion columns to avoid duplicates
        emotion_cols = ['emotion_label', 'emotion_confidence'] + EMOTIONS
        df_clean = df.drop(columns=[c for c in emotion_cols if c in df.columns], errors='ignore')
        # Concatenate results to original dataframe
        df_with_emotions = pd.concat([df_clean, emotion_results], axis=1)
        # Print distribution
        dist = df_with_emotions['emotion_label'].value_counts()
        print(f"    Emotion distribution:")
        for emotion, count in dist.items():
            pct = count / len(df_with_emotions) * 100
            print(f"      {emotion}: {count:,} ({pct:.1f}%)")
        updated_datasets[name] = df_with_emotions
    return updated_datasets


# =============================================================================
# WORD CLOUD GENERATION
# =============================================================================

def create_emotion_wordcloud(
    texts: List[str],
    emotion: str,
    dataset_name: str
) -> Optional[WordCloud]:
    """
    Create a word cloud for a specific emotion.
    
    Args:
        texts: List of text strings for this emotion
        emotion: Name of the emotion
        dataset_name: Name of the dataset (for labeling)
        
    Returns:
        WordCloud object, or None if no text data
    """
    combined_text = ' '.join(texts)
    
    if len(combined_text.strip()) == 0:
        return None
    
    colormap = EMOTION_COLORMAP.get(emotion, 'viridis')
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(combined_text)
    
    return wordcloud


def save_wordcloud(
    wordcloud: WordCloud,
    title: str,
    filepath: Path
) -> None:
    """
    Save a word cloud to a PNG file.
    
    Args:
        wordcloud: WordCloud object to save
        title: Title for the visualization
        filepath: Path to save the PNG file
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_wordclouds_for_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path
) -> List[str]:
    """
    Generate all word clouds for a single dataset.
    
    Creates:
    - One word cloud per emotion (7 files)
    - One combined word cloud for all emotions (1 file)
    
    Args:
        df: DataFrame with 'cleaned_text' and 'emotion_label' columns
        dataset_name: Name of the dataset
        output_dir: Directory to save PNG files
        
    Returns:
        List of saved filenames
    """
    saved_files = []
    emotions = sorted(df['emotion_label'].unique())

    text_cols=('body', 'text')
    text_col = next((c for c in text_cols if c in df.columns), None)
    
    # Create word cloud for each emotion
    for emotion in emotions:
        emotion_texts = df[df['emotion_label'] == emotion][text_col].fillna('').tolist()
        
        wordcloud = create_emotion_wordcloud(emotion_texts, emotion, dataset_name)
        
        if wordcloud is None:
            print(f"    WARNING: Skipping {emotion} - no text data")
            continue
        
        filename = f"{dataset_name.lower()}_{emotion}_wordcloud.png"
        filepath = output_dir / filename
        
        title = f'{dataset_name} - {emotion.capitalize()} Emotion Word Cloud'
        save_wordcloud(wordcloud, title, filepath)
        
        saved_files.append(filename)
        print(f"    Saved: {filename}")
    
    # Create combined word cloud
    text_cols=('body', 'text')
    text_col = next((c for c in text_cols if c in df.columns), None)
    all_texts = df[text_col].fillna('').tolist()
    combined_text = ' '.join(all_texts)
    
    if combined_text.strip():
        wordcloud_all = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(combined_text)
        
        filename = f"{dataset_name.lower()}_all_emotions_wordcloud.png"
        filepath = output_dir / filename
        
        title = f'{dataset_name} - All Emotions Combined'
        save_wordcloud(wordcloud_all, title, filepath)
        
        saved_files.append(filename)
        print(f"    Saved: {filename}")
    
    return saved_files


# =============================================================================
# TIME SERIES VISUALIZATION
# =============================================================================

def prepare_time_column(
    df: pd.DataFrame,
    dataset_name: str
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare datetime column based on dataset type.
    
    Twitter uses 'created_at' (datetime strings) with hourly aggregation.
    Reddit uses 'created_utc' (Unix timestamps) with monthly aggregation.
    
    Args:
        df: DataFrame to process
        dataset_name: 'Twitter' or 'Reddit'
        
    Returns:
        Tuple of (DataFrame with time_period column, time format string)
    """
    df_plot = df.copy()
    
    if dataset_name == 'Twitter':
        time_col = 'created_at'
        df_plot['datetime'] = pd.to_datetime(df_plot[time_col], errors='coerce')
        df_plot['time_period'] = df_plot['datetime'].dt.floor('H')
        time_format = '%Y-%m-%d %H:%M'
    else:  # Reddit
        time_col = 'created_utc'
        df_plot['datetime'] = pd.to_datetime(df_plot[time_col], unit='s', errors='coerce')
        df_plot['time_period'] = df_plot['datetime'].dt.to_period('M').dt.to_timestamp()
        time_format = '%Y-%m'
    
    return df_plot, time_format


def create_emotion_timeseries(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path
) -> List[str]:
    """
    Create interactive time-series visualizations of emotion trends.
    
    Generates two visualizations:
    1. Percentage view: Shows emotion distribution as % of posts over time
    2. Stacked count view: Shows absolute counts stacked by emotion
    
    Args:
        df: DataFrame with emotion data
        dataset_name: 'Twitter' or 'Reddit'
        output_dir: Directory to save HTML files
        
    Returns:
        List of saved filenames
    """
    saved_files = []
    
    # Prepare time column
    df_plot, time_format = prepare_time_column(df, dataset_name)
    
    # Count emotions by time period
    emotion_counts = (
        df_plot
        .groupby(['time_period', 'emotion_label'])
        .size()
        .reset_index(name='count')
    )
    
    # Calculate percentages
    total_by_period = emotion_counts.groupby('time_period')['count'].transform('sum')
    emotion_counts['percentage'] = (emotion_counts['count'] / total_by_period * 100).round(2)
    
    emotions = sorted(emotion_counts['emotion_label'].unique())
    
    # --- Percentage View ---
    fig_pct = go.Figure()
    
    for emotion in emotions:
        emotion_data = emotion_counts[emotion_counts['emotion_label'] == emotion]
        
        fig_pct.add_trace(go.Scatter(
            x=emotion_data['time_period'],
            y=emotion_data['percentage'],
            name=emotion.capitalize(),
            mode='lines+markers',
            line=dict(color=EMOTION_LINE_COLORS.get(emotion, '#000000'), width=2),
            marker=dict(size=6),
            hovertemplate=(
                f'<b>{emotion.capitalize()}</b><br>'
                'Date: %{x}<br>'
                'Percentage: %{y:.1f}%<br>'
                '<extra></extra>'
            )
        ))
    
    fig_pct.update_layout(
        title=dict(
            text=f'{dataset_name} - Emotion Trends Over Time',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title='Time Period',
        yaxis_title='Percentage of Posts (%)',
        hovermode='x unified',
        height=600,
        width=1200,
        template='plotly_white',
        legend=dict(
            title='Emotions',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        margin=dict(l=60, r=150, t=80, b=60)
    )
    
    filename_pct = f"{dataset_name.lower()}_emotion_timeseries.html"
    filepath_pct = output_dir / filename_pct
    fig_pct.write_html(str(filepath_pct))
    saved_files.append(filename_pct)
    print(f"    Saved: {filename_pct}")
    
    # --- Stacked Count View ---
    fig_count = go.Figure()
    
    for emotion in emotions:
        emotion_data = emotion_counts[emotion_counts['emotion_label'] == emotion]
        
        fig_count.add_trace(go.Scatter(
            x=emotion_data['time_period'],
            y=emotion_data['count'],
            name=emotion.capitalize(),
            mode='lines+markers',
            line=dict(color=EMOTION_LINE_COLORS.get(emotion, '#000000'), width=2),
            marker=dict(size=6),
            stackgroup='one',
            hovertemplate=(
                f'<b>{emotion.capitalize()}</b><br>'
                'Date: %{x}<br>'
                'Count: %{y}<br>'
                '<extra></extra>'
            )
        ))
    
    fig_count.update_layout(
        title=dict(
            text=f'{dataset_name} - Emotion Counts Over Time (Stacked)',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title='Time Period',
        yaxis_title='Number of Posts',
        hovermode='x unified',
        height=600,
        width=1200,
        template='plotly_white',
        legend=dict(
            title='Emotions',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        margin=dict(l=60, r=150, t=80, b=60)
    )
    
    filename_count = f"{dataset_name.lower()}_emotion_timeseries_stacked.html"
    filepath_count = output_dir / filename_count
    fig_count.write_html(str(filepath_count))
    saved_files.append(filename_count)
    print(f"    Saved: {filename_count}")
    
    return saved_files


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def create_summary_report(
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path
) -> str:
    """
    Create an HTML summary report of the emotion analysis.
    
    Args:
        datasets: Dictionary of processed DataFrames
        output_dir: Directory to save the report
        
    Returns:
        Path to saved report
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .dataset {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>Emotion Analysis Summary Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    for name, df in datasets.items():
        emotion_dist = df['emotion_label'].value_counts()
        total = len(df)
        
        html_content += f"""
        <div class="dataset">
            <h2>{name} Dataset</h2>
            <div class="metric">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Posts</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(emotion_dist)}</div>
                <div class="metric-label">Unique Emotions</div>
            </div>
            <div class="metric">
                <div class="metric-value">{emotion_dist.iloc[0]:,}</div>
                <div class="metric-label">Most Common: {emotion_dist.index[0].capitalize()}</div>
            </div>

            <h3>Emotion Distribution</h3>
            <table>
                <tr>
                    <th>Emotion</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for emotion, count in emotion_dist.items():
            percentage = count / total * 100
            html_content += f"""
                <tr>
                    <td><strong>{emotion.capitalize()}</strong></td>
                    <td>{count:,}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    html_content += """
        <div class="dataset">
            <h2>Generated Files</h2>
            <h3>Word Clouds (PNG)</h3>
            <ul>
                <li>Individual emotion word clouds for Twitter (8 files)</li>
                <li>Individual emotion word clouds for Reddit (8 files)</li>
            </ul>
            <h3>Time-Series (HTML - Interactive)</h3>
            <ul>
                <li>twitter_emotion_timeseries.html (percentage view)</li>
                <li>twitter_emotion_timeseries_stacked.html (count view)</li>
                <li>reddit_emotion_timeseries.html (percentage view)</li>
                <li>reddit_emotion_timeseries_stacked.html (count view)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    filepath = output_dir / 'emotion_analysis_summary.html'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(filepath)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline() -> None:
    """
    Execute the complete emotion visualization pipeline.
    
    Steps:
    1. Load environment and create directories
    2. Find and load CSV datasets
    3. Load emotion detection model
    4. Detect emotions for all texts
    5. Generate word clouds
    6. Generate time-series visualizations
    7. Create summary report
    """
    print("=" * 60)
    print(" EMOTION VISUALIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Environment setup
    print("\n[1/7] Setting up environment...")
    env = load_environment()
    directories = create_directories(env['output_dir'])
    print(f"  Data directory: {env['data_dir']}")
    print(f"  Output directory: {env['output_dir']}")
    
    # Step 2: Find and load data
    print("\n[2/7] Loading datasets...")
    try:
        file_paths = find_csv_files(env['data_dir'])
        datasets = load_datasets(file_paths)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nERROR: Error loading data: {e}")
        #return
        raise SystemExit(1)

    
    # Step 3: Load emotion model
    print("\n[3/7] Loading emotion detection model...")
    emotion_classifier, device_name = load_emotion_model()
    print(f"  Model loaded successfully on {device_name}")
    
    # Step 4: Detect emotions
    print("\n[4/7] Detecting emotions...")
    print("  This may take several minutes depending on dataset size...")
    datasets = add_emotions_to_datasets(datasets, emotion_classifier)
    print("  Emotion detection complete")
    
    # Step 5: Generate word clouds
    print("\n[5/7] Generating word clouds...")
    wordcloud_dir = directories['wordclouds']
    all_wordcloud_files = []
    
    for name, df in datasets.items():
        print(f"\n  Creating word clouds for {name}...")
        files = generate_wordclouds_for_dataset(df, name, wordcloud_dir)
        all_wordcloud_files.extend(files)
    
    print(f"  Generated {len(all_wordcloud_files)} word cloud files")
    
    # Step 6: Generate time-series
    print("\n[6/7] Generating time-series visualizations...")
    timeseries_dir = directories['timeseries']
    all_timeseries_files = []
    
    for name, df in datasets.items():
        print(f"\n  Creating time-series for {name}...")
        files = create_emotion_timeseries(df, name, timeseries_dir)
        all_timeseries_files.extend(files)
    
    print(f"  Generated {len(all_timeseries_files)} time-series files")
    
    # Step 7: Create summary report
    print("\n[7/7] Creating summary report...")
    summary_path = create_summary_report(datasets, directories['emotions'])
    print(f"  Summary report saved: {summary_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {env['output_dir']}")
    print(f"\nGenerated files:")
    print(f"  Word Clouds: {len(all_wordcloud_files)} PNG files")
    print(f"  Time-Series: {len(all_timeseries_files)} HTML files")
    print(f"  Summary Report: 1 HTML file")
    print(f"\nTotal files generated: {len(all_wordcloud_files) + len(all_timeseries_files) + 1}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()