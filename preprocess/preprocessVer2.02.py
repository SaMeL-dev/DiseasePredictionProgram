import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv("BRFSS_2015ver14.csv")


df_info = pd.DataFrame({
    'dtype': df.dtypes,
    'sample_values': df.apply(lambda col: col.dropna().unique()[:10])
})


print(df_info)

