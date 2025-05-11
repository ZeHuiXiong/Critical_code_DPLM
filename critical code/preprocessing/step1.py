import pandas as pd
import re
import os

file_path = 'TDMs.xlsx'
file_path_mid = 'TDMs_formula.xlsx'
file_path_MSF = 'descriptor_bear.xlsx'
file_path_out = 'TDMs_s1.xlsx'

df = pd.read_excel(file_path)
def parse_formula(formula):
    elements = re.findall(r'[A-Z][a-z]?', formula)
    elements += [None] * (4 - len(elements))
    return elements

parsed_df = df['formula'].apply(parse_formula).apply(pd.Series)
parsed_df.columns = ['Element_A', 'Element_B1', 'Element_B2', 'Element_X']
df = pd.concat([df, parsed_df], axis=1)
selected_columns = ['formula', 'Element_A', 'Element_B1', 'Element_B2', 'Element_X']
df.to_excel(file_path_mid, index=False)
df = pd.read_excel(file_path_mid)

def generate_formula(row):
    col1_clean = row['Element_A'].strip()
    col2_clean = row['Element_B1'].strip()
    col3_clean = row['Element_B2'].strip()
    col4_clean = row['Element_X'].strip()
    formula = f"{col1_clean}8 {col2_clean}4 {col3_clean}4 {col4_clean}24"
    return formula

df['formula'] = df.apply(generate_formula, axis=1)
df['formula'] = df['formula'].apply(lambda x: x.strip())
df = df[['formula'] + [col for col in df.columns if col != 'formula']]
df.to_excel(file_path_mid, index=False)

df = pd.read_excel(file_path)
dg = pd.read_excel(file_path_MSF)
li1 = []
li2 = []
li3 = []
li0 = []
for index, row in df.iterrows():
 temString = pd.DataFrame(row).iloc[0]
 ResString = temString.item().split(' ')
 li0.append(ResString[0])
 li1.append(ResString[1])
 li2.append(ResString[2])
 li3.append(ResString[3])

df['Element_A'] = pd.DataFrame(li0)
df['Element_B1'] = pd.DataFrame(li1)
df['Element_B2'] = pd.DataFrame(li2)
df['Element_X'] = pd.DataFrame(li3)

def add_feature(df,dg,element_type):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []

    for z in element_type:
        str_name = 'Element_' + z
        for i in range(len(df)):
            element_name = df[str_name][i]
            tag = 1
            for j in range(len(dg)):
                element = dg['element'][j]
                if element_name == element:
                    tag = 0
                    list1.append(dg['Number'][j])
                    list2.append(dg['Relative mess'][j])
                    list3.append(dg['Volume'][j])
                    list4.append(dg['Density'][j])
                    list5.append(dg['Atomic radius (A)'][j])
                    list6.append(dg['Covalent radius (A)'][j])
                    list7.append(dg['Effective ion radius'][j])

            if tag:
                print(element_name)
        col1 = z+'_' + 'Number'
        df[col1] = list1
        col2 = z+'_' +'Relative mess'
        df[col2] = list2
        col3 = z+'_' +'Volume'
        df[col3] = list3
        col4 = z+'_' +'Density'
        df[col4] = list4
        col5 = z+'_' +'Atomic radius (A)'
        df[col5] = list5
        col6 = z+'_' +'Covalent radius (A)'
        df[col6] = list6
        col7 = z+'_' +'Effective ion radius'
        df[col7] = list7

        list1.clear()
        list2.clear()
        list3.clear()
        list4.clear()
        list5.clear()
        list6.clear()
        list7.clear()

    return df

result = add_feature(df,dg,['A','B1','B2','X'])
result = result.drop(columns=['Element_A', 'Element_B1', 'Element_B2', 'Element_X'])
print(result)
os.remove(file_path_mid)
result.to_excel(file_path_out,index = False)