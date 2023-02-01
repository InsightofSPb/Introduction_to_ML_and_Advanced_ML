import pandas as pd
import matplotlib.pyplot as plt


ap = pd.read_csv('Pulsar.csv')
p1 = ap[((ap.TG == 0) & (94.6640625 <= ap.MIP) & (ap.TG == 0) & (ap.MIP <= 95.2890625)) | ((ap.TG == 1) & (65.078125 <= ap.MIP) & (ap.TG == 1) & (ap.MIP <= 70.7421875))]

w = p1.MIP.mean()
print(p1.shape)
print(w)


zp = pd.read_csv('Zp.csv', delimiter=',', index_col='REGION_NAME')
zp = zp.drop(['Курская область'])
zp = zp.drop(['Республика Мордовия'])
zp = zp.sort_values('SALARY')
sr = zp.SALARY.mean()
med = zp.SALARY.median()
print(zp)
print('X17:', zp.SALARY[16], 'X40:', zp.SALARY[39], 'X41:', zp.SALARY[40])
print('Среднее:', round(sr, 2), 'Медиана:', med, sep='\n')