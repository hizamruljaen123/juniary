#!/usr/bin/env python3
"""
Test script untuk mengecek apakah fix database connection sudah bekerja
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_analyzer():
    """Test analyzer dengan fix database connection"""
    print("=== Testing Fixed Analyzer ===")
    try:
        from db_manager import DatabaseManager
        
        # Create a simplified analyzer like the fixed version
        class FixedSentimentAnalyzer:
            def __init__(self):
                # Initialize database manager and connect
                self.db_manager = DatabaseManager()
                self.db_connected = self.db_manager.connect()
                
                if self.db_connected:
                    print("✅ Database connection successful for SentimentAnalyzer")
                else:
                    print("⚠️ Warning: Database connection failed. Using CSV fallback.")
        
        print("1. Creating fixed analyzer...")
        analyzer = FixedSentimentAnalyzer()
        
        # Test the FIXED condition from process_dataset
        use_database = (hasattr(analyzer, 'db_manager') and 
                       hasattr(analyzer, 'db_connected') and 
                       analyzer.db_connected and 
                       analyzer.db_manager.connection)
        
        print(f"2. hasattr(analyzer, 'db_manager'): {hasattr(analyzer, 'db_manager')}")
        print(f"3. hasattr(analyzer, 'db_connected'): {hasattr(analyzer, 'db_connected')}")
        print(f"4. analyzer.db_connected: {analyzer.db_connected}")
        print(f"5. analyzer.db_manager.connection: {analyzer.db_manager.connection}")
        print(f"6. use_database final result: {use_database}")
        print(f"7. use_database type: {type(use_database)}")
        
        if use_database:
            print("✅ Database AKAN digunakan untuk menyimpan data!")
            
            # Test actual saving
            success = analyzer.db_manager.save_sentiment_data(
                text="Test fixed analyzer",
                preprocessed_text="test fixed analyzer",
                sentiment="positive",
                confidence=0.95,
                favorite_count=7,
                source_dataset="fixed_test"
            )
            
            if success:
                print("✅ Data berhasil disimpan dengan analyzer yang sudah diperbaiki!")
            else:
                print("❌ Gagal menyimpan data")
        else:
            print("❌ Database TIDAK akan digunakan")
            print("   Data akan disimpan ke CSV saja")
            
        analyzer.db_manager.disconnect()
        
    except Exception as e:
        print(f"❌ Error testing fixed analyzer: {e}")
        import traceback
        traceback.print_exc()

def test_app_condition():
    """Test kondisi yang sama persis seperti di app.py setelah fix"""
    print("\n=== Testing App.py Fixed Condition ===")
    try:
        # Simulate the exact condition that will be used in app.py after fix
        from db_manager import DatabaseManager
        
        # Mock analyzer with the fixed attributes
        class MockFixedAnalyzer:
            def __init__(self):
                self.db_manager = DatabaseManager()
                self.db_connected = self.db_manager.connect()
        
        analyzer = MockFixedAnalyzer()
        
        # Test the EXACT condition from the fixed app.py
        condition1 = hasattr(analyzer, 'db_manager')
        condition2 = hasattr(analyzer, 'db_connected') 
        condition3 = analyzer.db_connected
        condition4 = analyzer.db_manager.connection
        
        print(f"Condition 1 - hasattr(analyzer, 'db_manager'): {condition1}")
        print(f"Condition 2 - hasattr(analyzer, 'db_connected'): {condition2}")  
        print(f"Condition 3 - analyzer.db_connected: {condition3}")
        print(f"Condition 4 - analyzer.db_manager.connection: {condition4}")
        
        use_database = condition1 and condition2 and condition3 and condition4
        print(f"\nFinal use_database: {use_database} (type: {type(use_database)})")
        
        if use_database is True:
            print("✅ PERFECT! Database akan digunakan dan kondisinya boolean True")
        elif use_database:
            print(f"⚠️ Database akan digunakan tapi kondisinya bukan boolean: {type(use_database)}")
        else:
            print("❌ Database tidak akan digunakan")
            
        analyzer.db_manager.disconnect()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_analyzer()
    test_app_condition()
