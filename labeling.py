import pandas as pd
import numpy as np
import re
import json
import io
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

def analyze_text_batch(analyzer, texts, favorite_counts=None, max_workers=4):
    """
    Analyze a batch of texts in parallel using ThreadPoolExecutor.
    
    Args:
        analyzer: The SentimentAnalyzer instance
        texts: List of texts to analyze
        favorite_counts: Optional list of favorite counts (same length as texts)
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of analysis results
    """
    if favorite_counts is None:
        favorite_counts = [0] * len(texts)
    
    results = []
    progress_queue = Queue()
    
    def analyze_single_text(idx, text, favorite_count):
        try:
            # Skip empty or NaN texts
            if pd.isna(text) or text == '':
                return idx, None
            
            # Analyze the text
            analysis = analyzer.analyze_text(text)
            
            # Apply rules if available
            extended_analysis = analyzer.apply_rules(analysis, favorite_count)
            analysis.update(extended_analysis)
            
            # Return the result with its index
            return idx, analysis
        except Exception as e:
            return idx, {"error": str(e)}
    
    # Create and submit tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_single_text, i, text, favorite_counts[i]): i 
            for i, text in enumerate(texts) if not pd.isna(text) and text != ''
        }
        
        # Initialize results with None placeholders
        results = [None] * len(texts)
        
        # Process results as they complete
        for future in future_to_idx:
            try:
                idx, result = future.result()
                if result is not None:
                    results[idx] = result
            except Exception as e:
                # Handle any unexpected errors
                print(f"Error in analyze_text_batch: {e}")
    
    return results

def stream_process_dataset(analyzer, input_df, progress_callback, save_to_db=True):
    """
    Process the dataset and apply sentiment analysis to each row.
    Optionally save to database during processing.
    
    Args:
        analyzer: The SentimentAnalyzer instance
        input_df: The input DataFrame to process
        progress_callback: Callback function to report progress
        save_to_db: Whether to save to database during processing
        
    Returns:
        Processed DataFrame with sentiment analysis results
    """
    total_rows = len(input_df)
    batch_size = min(100, max(10, total_rows // 20))  # Adjust batch size based on total rows
    
    # Initialize output data
    labeled_data = []
    saved_to_db_count = 0
    
    # Check if database is available
    use_database = save_to_db and hasattr(analyzer, 'db_manager') and analyzer.db_manager.connection
    
    # Report start
    progress_callback(0, "Memulai proses analisis sentiment...")
    
    # Process in batches for better performance and progress tracking
    for i in range(0, total_rows, batch_size):
        batch_df = input_df.iloc[i:i+batch_size].copy()
        
        # Extract text and favorite count
        texts = batch_df['full_text'].tolist()
        favorite_counts = [0] * len(texts)  # Default to 0
        
        # Use favorite_count if available
        if 'favorite_count' in batch_df.columns:
            favorite_counts = batch_df['favorite_count'].fillna(0).tolist()
        
        # Process batch in parallel
        progress_callback(int(i/total_rows*80), f"Menganalisis batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size}...")
        batch_results = analyze_text_batch(analyzer, texts, favorite_counts)
        
        # Process each row in the batch
        for j, row in batch_df.iterrows():
            idx = j - i  # Index within the batch
            result = batch_results[idx]
            
            if result is None or 'error' in result:
                progress_callback(int(i/total_rows*80), f"Melewati baris {j+1}: Teks kosong atau error", "error")
                continue
            
            # Create entry with default values
            entry = {
                'text': row['full_text'],
                'preprocessed_text': result['text'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'favorite_count': int(row.get('favorite_count', 0)) if pd.notna(row.get('favorite_count', 0)) else 0,
                'created_at': row.get('created_at') if pd.notna(row.get('created_at')) else None,
                'location': row.get('location') if pd.notna(row.get('location')) else None,
                'username': row.get('username') if pd.notna(row.get('username')) else None,
                'source_dataset': 'uploaded_data'
            }
            
            # Add optional fields if available
            for field in ['favorite_count', 'created_at', 'location', 'username']:
                if field in row and not pd.isna(row[field]):
                    if field == 'favorite_count':
                        entry[field] = int(row[field]) if row[field] != '' else 0
                    else:
                        entry[field] = row[field]
            
            # Save to database immediately if available
            if use_database:
                try:
                    success = analyzer.db_manager.save_sentiment_data(
                        text=entry['text'],
                        preprocessed_text=entry['preprocessed_text'],
                        sentiment=entry['sentiment'],
                        confidence=entry['confidence'],
                        favorite_count=entry['favorite_count'],
                        created_at=entry['created_at'],
                        location=entry['location'],
                        username=entry['username'],
                        source_dataset=entry['source_dataset']
                    )
                    if success:
                        saved_to_db_count += 1
                except Exception as e:
                    progress_callback(int(i/total_rows*80), f"Error saving to database: {str(e)}", "error")
            
            labeled_data.append(entry)
            
            # Report individual progress occasionally
            if (j - i) % 10 == 0 or j == i + batch_size - 1:
                if use_database:
                    progress_callback(int((i + j - i + 1)/total_rows*80), f"Diproses: {j+1}/{total_rows} baris, tersimpan ke DB: {saved_to_db_count}")
                else:
                    progress_callback(int((i + j - i + 1)/total_rows*80), f"Diproses: {j+1}/{total_rows} baris")
    
    # Convert to DataFrame
    df_labeled = pd.DataFrame(labeled_data)
    
    # Report completion
    if use_database:
        progress_callback(100, f"Analisis selesai! Total {len(df_labeled)} baris berhasil dilabeli, {saved_to_db_count} tersimpan ke database.", "success")
    else:
        progress_callback(100, f"Analisis selesai! Total {len(df_labeled)} baris berhasil dilabeli.", "success")
    
    return df_labeled
