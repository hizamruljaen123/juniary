#!/usr/bin/env python3
"""
Test script untuk mengecek koneksi database dan memperbaiki masalah penyimpanan data
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """Test koneksi ke database MySQL"""
    try:
        from db_manager import DatabaseManager
        
        print("Testing database connection...")
        db = DatabaseManager()
        
        # Test connection
        if db.connect():
            print("✅ Database connection successful!")
            
            # Test if connection is active
            if db.connection and db.connection.is_connected():
                print("✅ Database connection is active")
                
                # Test basic query
                cursor = db.execute_query("SELECT 1 as test")
                if cursor:
                    result = cursor.fetchone()
                    cursor.close()
                    print(f"✅ Basic query successful: {result}")
                    
                    # Test if tables exist
                    cursor = db.execute_query("SHOW TABLES")
                    if cursor:
                        tables = [row['Tables_in_sentiment_data'] for row in cursor.fetchall()]
                        cursor.close()
                        print(f"✅ Found tables: {tables}")
                        
                        # Test inserting sample data
                        success = db.save_sentiment_data(
                            text="Test text",
                            preprocessed_text="test text",
                            sentiment="positive",
                            confidence=0.85,
                            favorite_count=10,
                            source_dataset="test_data"
                        )
                        
                        if success:
                            print("✅ Sample data insertion successful!")
                        else:
                            print("❌ Sample data insertion failed!")
                            
                    else:
                        print("❌ Failed to show tables")
                else:
                    print("❌ Basic query failed")
            else:
                print("❌ Database connection is not active")
        else:
            print("❌ Database connection failed!")
            
        db.disconnect()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install mysql-connector-python pandas")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_analyzer_initialization():
    """Test inisialisasi analyzer dengan database"""
    try:
        from labeling import SentimentAnalyzer
        
        print("\nTesting analyzer initialization...")
        analyzer = SentimentAnalyzer()
        
        if hasattr(analyzer, 'db_manager') and analyzer.db_manager:
            print("✅ Analyzer has db_manager")
            
            if analyzer.db_manager.connection:
                print("✅ Analyzer db_manager has active connection")
            else:
                print("❌ Analyzer db_manager connection is None")
                # Try to reconnect
                if analyzer.db_manager.connect():
                    print("✅ Reconnected db_manager successfully")
                else:
                    print("❌ Failed to reconnect db_manager")
        else:
            print("❌ Analyzer does not have db_manager")
            
    except Exception as e:
        print(f"❌ Error testing analyzer: {e}")

if __name__ == "__main__":
    test_database_connection()
    test_analyzer_initialization()
