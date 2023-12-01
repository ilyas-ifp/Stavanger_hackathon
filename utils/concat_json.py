import os
import pandas as pd
import json
def concat_json():
    # Specify the folder where your JSON files are located
    folder_path = "./json/updated"
    # Create an empty list to store DataFrames
    dataframes_list = []
    # Loop through JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                df = pd.DataFrame.from_dict(data)
                dataframes_list.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes_list, ignore_index=True)
    return combined_df
# Now combined_df contains the data from all JSON files

if __name__ == '__main__' :
    df = concat_json()
    print(df)