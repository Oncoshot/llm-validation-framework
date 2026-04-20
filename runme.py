import os
import llmvalidate.validation as v
import pandas as pd
pd.options.display.width = 0
from datetime import datetime


def main():
    input_file  = r"samples.csv"
    source_df = pd.read_csv(input_file, index_col='Patient ID')

    fields=None #['First Primary Diagnosis','First Primary Histology','Treatment Drugs']
    
    # Extract input file name without extension and create subfolder
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join("validation_results", input_filename)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    file_prefix = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    res_df, metrics_df = v.validate(
                                        source_df, 
                                        fields, 
                                        file_prefix = file_prefix,
                                        structure_callback=None, 
                                        raw_text_column_name = None, 
                                        output_folder=output_folder,
                                        max_workers= 1, 
                                        use_threads =True)
    print(f"")
    print(f"Validation is Completed!")
    print(f"Validation Results and Metrics are Saved in: {output_folder}")


    print(f"")
    print(f"Calculating Confidence Intervals using Bootstrapping...")

    # Bootstrap_CI doesnt support partially labeled datasets, so we need to drop rows without labels in the specified fields
    # Let's do bootstrapping only on the rows with 'First Primary Diagnosis' and 'First Primary Histology' labels
    # You can change this to 'Has metastasis' or 'Treatment Drugs'/'Test Results'
    fields = ['First Primary Diagnosis','First Primary Histology']
    res_df = res_df.dropna(subset=fields, how='any')

    ci_df = v.bootstrap_CI(res_df, fields, n_bootstrap=100, ci=0.95, random_state=42)

   # Save metrics
    if output_folder:
        ci_df.to_csv(os.path.join(output_folder, f"{file_prefix} CI metrics.csv"), index=False)

    print(f"Confidence Intervals are Saved in: {output_folder}")

if __name__ == "__main__":                           # critical on Windows
    main()