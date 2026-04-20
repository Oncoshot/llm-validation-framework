import pandas as pd
import os

from pandas import DataFrame


def read_parents():
    datalists = ['Diagnoses_Cancer','Diagnoses_NonCancer',
                 'CurrentMedicationTreatment_Cancer','CurrentMedicationTreatment_NonCancer',
                 'Biomarkers']
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Loop through each file name in the list
    for datalist in datalists:
        # Construct the file path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', f'{datalist}.csv')

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, usecols=['id', 'pid', 'value'])

        # Add the 'domain_datalist' column
        df['domain_datalist'] = datalist

        # Append the DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        #print(combined_df)

        # Self-join on df.pid = df.id
    joined_df = pd.merge(combined_df, combined_df, left_on=['pid', 'domain_datalist'], right_on=['id', 'domain_datalist'],
                         suffixes=('_child', '_parent'))

    # Create dictionary:
    parents = dict(zip(joined_df['value_child'], joined_df['value_parent']))

    return parents

def get_datalist(datalist:str) -> DataFrame:
    # Construct the file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', f'{datalist}.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=['id', 'pid', 'value'])

    return df

def get_parents(datalist:str):
    # Construct the file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', f'{datalist}.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=['id', 'pid', 'value'])

    # extract only parent items
    parents = df[df['pid'].isnull()][['value']]

    return parents

# Get children with specific parent from data list
def get_children(datalist:str, parent:str):
    # Construct the file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', f'{datalist}.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=['id', 'pid', 'value'])

    parent_id = df[df['value'] == parent].iloc[0]['id']

    # extract only items with particular parent
    children = df[df['pid'] == parent_id][['value']]

    return children

# check if there is exact case insensitive match among all datalist items
# or items without parent (if childOnly=True)
def datalist_contains_value(datalist: DataFrame, diagnosis:str, childOnly=False) -> bool:
    items = datalist[datalist['value'].str.lower() == diagnosis.lower()]

    if childOnly:
        items = items[items['pid'].notna()]

    return items.shape[0] > 0