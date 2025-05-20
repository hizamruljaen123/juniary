-- Schema untuk database sentiment_data

-- Create database jika belum ada
CREATE DATABASE IF NOT EXISTS sentiment_data;

-- Gunakan database
USE sentiment_data;

-- Tabel untuk menyimpan data teks
CREATE TABLE IF NOT EXISTS texts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    preprocessed_text TEXT NOT NULL,
    created_at DATETIME,
    source_dataset VARCHAR(100)
);

-- Tabel untuk menyimpan hasil analisis sentimen
CREATE TABLE IF NOT EXISTS sentiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text_id INT NOT NULL,
    sentiment ENUM('positive', 'negative', 'neutral') NOT NULL,
    confidence FLOAT NOT NULL,
    FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
);

-- Tabel untuk menyimpan metadata tambahan
CREATE TABLE IF NOT EXISTS metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text_id INT NOT NULL,
    favorite_count INT DEFAULT 0,
    location VARCHAR(255),
    username VARCHAR(100),
    FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
);

-- Tabel untuk menyimpan rules yang dihasilkan model
CREATE TABLE IF NOT EXISTS rules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rule_type VARCHAR(50) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    rule_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabel untuk menyimpan model
CREATE TABLE IF NOT EXISTS models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_data LONGBLOB,
    vectorizer_data LONGBLOB,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabel untuk tracking dataset splits (untuk training)
CREATE TABLE IF NOT EXISTS dataset_splits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text_id INT NOT NULL,
    split_type ENUM('train', 'test') NOT NULL,
    split_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
);

-- View untuk memudahkan query data lengkap
CREATE OR REPLACE VIEW labeled_data_view AS
SELECT 
    t.id,
    t.text,
    t.preprocessed_text,
    s.sentiment,
    s.confidence,
    m.favorite_count,
    t.created_at,
    m.location,
    m.username,
    t.source_dataset
FROM 
    texts t
JOIN 
    sentiments s ON t.id = s.text_id
LEFT JOIN 
    metadata m ON t.id = m.text_id;

-- Indeks untuk meningkatkan performa query
CREATE INDEX idx_sentiment ON sentiments(sentiment);
CREATE INDEX idx_source_dataset ON texts(source_dataset);
