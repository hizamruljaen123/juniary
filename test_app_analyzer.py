#!/usr/bin/env python3
"""
Test script baru untuk mengecek masalah database connection pada analyzer
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direct_analyzer():
    """Test analyzer secara langsung tanpa import dari app.py"""
    print("=== Testing Direct Analyzer Creation ===")
    try:
        # Import components directly
        from db_manager import DatabaseManager
        
        # Create database manager
        print("1. Creating database manager...")
        db_manager = DatabaseManager()
        
        # Test connection
        print("2. Testing database connection...")
        if db_manager.connect():
            print("✅ Database connected successfully")
            
            # Test if connection is active
            if db_manager.connection and db_manager.connection.is_connected():
                print("✅ Database connection is active")
                
                # Create a simple analyzer-like object
                class SimpleAnalyzer:
                    def __init__(self, db_manager):
                        self.db_manager = db_manager
                
                analyzer = SimpleAnalyzer(db_manager)
                print("✅ Simple analyzer created with db_manager")
                
                # Test the condition that's used in app.py
                use_database = hasattr(analyzer, 'db_manager') and analyzer.db_manager.connection
                print(f"✅ use_database condition: {use_database}")
                
                if use_database:
                    print("✅ Database will be used for saving data")
                    
                    # Test actual save
                    success = analyzer.db_manager.save_sentiment_data(
                        text="Test from analyzer",
                        preprocessed_text="test from analyzer",
                        sentiment="positive",
                        confidence=0.90,
                        favorite_count=5,
                        source_dataset="analyzer_test"
                    )
                    
                    if success:
                        print("✅ Data saved successfully via analyzer!")
                    else:
                        print("❌ Failed to save data via analyzer")
                else:
                    print("❌ Database will NOT be used")
                    
            else:
                print("❌ Database connection is not active")
        else:
            print("❌ Failed to connect to database")
            
        db_manager.disconnect()
        
    except Exception as e:
        print(f"❌ Error in direct analyzer test: {e}")
        import traceback
        traceback.print_exc()

def test_app_analyzer():
    """Test analyzer dari app.py secara langsung"""
    print("\n=== Testing App.py Analyzer ===")
    try:
        # Import minimal components needed
        import pandas as pd
        from flask import Flask
        
        # Create minimal flask app context
        app = Flask(__name__)
        app.secret_key = 'test_key'
        
        with app.app_context():
            # Now try to import and create the analyzer from app.py
            print("1. Importing from app.py...")
            
            # Read app.py and find the SentimentAnalyzer class definition
            with open('app.py', 'r', encoding='utf-8') as f:
                app_content = f.read()
                
            if 'class SentimentAnalyzer' in app_content:
                print("✅ Found SentimentAnalyzer class in app.py")
                
                # Try to execute the class definition
                exec_globals = {
                    'DatabaseManager': None,
                    'pd': pd,
                    'os': os,
                    'pickle': None,
                    'generate_rules_from_data': None
                }
                
                # Import DatabaseManager
                from db_manager import DatabaseManager
                exec_globals['DatabaseManager'] = DatabaseManager
                
                # Extract and execute just the SentimentAnalyzer class
                lines = app_content.split('\n')
                class_start = None
                class_lines = []
                indent_level = None
                
                for i, line in enumerate(lines):
                    if 'class SentimentAnalyzer' in line and line.strip().startswith('class'):
                        class_start = i
                        indent_level = len(line) - len(line.lstrip())
                        class_lines.append(line)
                        continue
                    
                    if class_start is not None:
                        if line.strip() == '' or line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                            class_lines.append(line)
                        else:
                            # End of class
                            break
                
                if class_lines:
                    class_code = '\n'.join(class_lines)
                    print("2. Executing SentimentAnalyzer class...")
                    
                    try:
                        exec(class_code, exec_globals)
                        SentimentAnalyzer = exec_globals['SentimentAnalyzer']
                        
                        print("3. Creating SentimentAnalyzer instance...")
                        analyzer = SentimentAnalyzer()
                        
                        print(f"✅ Analyzer created: {type(analyzer)}")
                        
                        # Check if it has db_manager
                        if hasattr(analyzer, 'db_manager'):
                            print(f"✅ Analyzer has db_manager: {type(analyzer.db_manager)}")
                            
                            if analyzer.db_manager and analyzer.db_manager.connection:
                                print("✅ Analyzer db_manager has active connection")
                                
                                # Test the exact condition from app.py
                                use_database = hasattr(analyzer, 'db_manager') and analyzer.db_manager.connection
                                print(f"✅ use_database (app.py condition): {use_database}")
                                
                            else:
                                print("❌ Analyzer db_manager has no connection")
                                print(f"   db_manager: {analyzer.db_manager}")
                                if analyzer.db_manager:
                                    print(f"   connection: {analyzer.db_manager.connection}")
                        else:
                            print("❌ Analyzer has no db_manager attribute")
                            
                    except Exception as e:
                        print(f"❌ Error executing class: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ Could not extract SentimentAnalyzer class")
            else:
                print("❌ SentimentAnalyzer class not found in app.py")
                
    except Exception as e:
        print(f"❌ Error in app analyzer test: {e}")
        import traceback
        traceback.print_exc()

def test_process_dataset_condition():
    """Test kondisi spesifik yang digunakan di process_dataset endpoint"""
    print("\n=== Testing Process Dataset Condition ===")
    try:
        from db_manager import DatabaseManager
        
        # Simulate the analyzer object that would be created in app.py
        class MockAnalyzer:
            def __init__(self):
                self.db_manager = DatabaseManager()
                # Try to connect
                self.db_manager.connect()
        
        print("1. Creating mock analyzer...")
        analyzer = MockAnalyzer()
        
        # Test the exact condition from process_dataset function
        use_database = hasattr(analyzer, 'db_manager') and analyzer.db_manager.connection
        
        print(f"2. hasattr(analyzer, 'db_manager'): {hasattr(analyzer, 'db_manager')}")
        print(f"3. analyzer.db_manager: {analyzer.db_manager}")
        print(f"4. analyzer.db_manager.connection: {analyzer.db_manager.connection}")
        print(f"5. use_database final result: {use_database}")
        
        if use_database:
            print("✅ Database akan digunakan untuk menyimpan data!")
            
            # Test saving
            success = analyzer.db_manager.save_sentiment_data(
                text="Test process dataset",
                preprocessed_text="test process dataset",
                sentiment="neutral",
                confidence=0.75,
                favorite_count=3,
                source_dataset="process_test"
            )
            
            if success:
                print("✅ Data berhasil disimpan melalui kondisi process_dataset!")
            else:
                print("❌ Gagal menyimpan data")
        else:
            print("❌ Database TIDAK akan digunakan")
            print("   Data akan disimpan ke CSV saja")
            
        analyzer.db_manager.disconnect()
        
    except Exception as e:
        print(f"❌ Error testing process dataset condition: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_analyzer()
    test_app_analyzer() 
    test_process_dataset_condition()
