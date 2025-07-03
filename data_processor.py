# data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    def __init__(self, config_file=None):
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file):
        # Load configuration
        if config_file and Path(config_file).exists():
            return pd.read_json(config_file)
        return {"default_settings": True}
    
    def process_file(self, input_path, output_path=None):
        """Process a single data file"""
        try:
            df = pd.read_csv(input_path)
            
            # Your processing logic here
            processed = self.clean_data(df)
            processed = self.analyze_data(processed)
            
            if output_path:
                processed.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
            
            return processed
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None
    
    def clean_data(self, df):
        """Data cleaning operations"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='forward')
        
        return df
    
    def analyze_data(self, df):
        """Data analysis operations"""
        # Add calculated columns
        if 'value' in df.columns:
            df['value_squared'] = df['value'] ** 2
            df['value_log'] = np.log(df['value'] + 1)
        
        return df
    
    def batch_process(self, input_dir, output_dir):
        """Process multiple files"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        for file_path in input_path.glob("*.csv"):
            output_file = output_path / f"processed_{file_path.name}"
            result = self.process_file(file_path, output_file)
            if result is not None:
                results.append(result)
        
        return results

def main():
    """Main application entry point"""
    processor = DataProcessor()
    
    # Process single file
    result = processor.process_file("input_data.csv", "output_data.csv")
    
    # Or batch process
    # results = processor.batch_process("data/raw/", "data/processed/")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
