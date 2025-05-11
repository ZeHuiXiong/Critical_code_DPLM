import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

if __name__ == '__main__':

    excel_file_path = 'TDMs_s1.xlsx'
    df = pd.read_excel(excel_file_path)

    print(df.head())
    data = StrToComposition().featurize_dataframe(df, 'formula')
    ep_feat = ElementProperty.from_preset(preset_name = 'magpie')
    data = ep_feat.featurize_dataframe(data,col_id = 'composition')

    data = data.drop(columns=['composition'])
    data.columns = data.columns.str.replace('MagpieData ', '', regex=False)

    data.to_excel('TDMs_all.xlsx', index=False)