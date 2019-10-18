# %% 
import pandas as pd 
 
data = pd.read_csv("../../bases/within_BE_dist.csv")
BE = data.BE_id.unique().tolist()


D = {v:[] for v in ['BE_id','com','next_com','km']}
for be in BE:
    data_be = data.loc[data.BE_id == be]
    codes = data_be.code_insee.unique().tolist()
    for code in codes:
        for next_code in codes:
            if code != next_code:
                comb = list(zip(D['com'],D['next_com']))
                if (code,next_code) not in comb and (next_code,code) not in comb : 
                    row = data_be.loc[data_be.code_insee == code].loc[data_be.code_insee_next == next_code]
                    D['BE_id'].append(row.iloc[0].BE_id)
                    D['com'].append(row.iloc[0].code_insee)
                    D['next_com'].append(row.iloc[0].code_insee_next)
                    D['km'].append(row.iloc[0].km)

pd.DataFrame(D).to_csv("../../bases/within_BE_dist_light.csv",index = False) 