import pandas as pd 
wtd_df = pd.read_csv('wtd_with_probing.csv')
wtd_df['message_id'] = ['msgid' + str(i) for i in range(len(wtd_df))]
wtd_df.to_csv('wtd_with_probing.csv')