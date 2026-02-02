import os
import src.validation as v
import pandas as pd
pd.options.display.width = 0


def main():
    input_file  = r"samples.csv"
    source_df = pd.read_csv(input_file, index_col='Patient ID')

    fields=None #['First Primary Diagnosis','First Primary Histology','Treatment Drugs']
    
    # Extract input file name without extension and create subfolder
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = ".\\validation_results\\" + input_filename
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    res_df, metrics_df = v.validate(
                                        source_df, 
                                        fields, 
                                        structure_callback=None, 
                                        raw_text_column_name = None, 
                                        output_folder=output_folder,
                                        max_workers= 1, 
                                        use_threads =True)
    print(f"")
    print(f"Validation is Completed!")
    print(f"Validation Results and Metrics are Saved in: {output_folder}")

if __name__ == "__main__":                           # critical on Windows
    main()