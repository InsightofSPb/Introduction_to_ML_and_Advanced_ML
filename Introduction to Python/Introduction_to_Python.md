## Задание 1
### Описание
Набор данных pulsar_stars  содержит сведения о звездах, полученные в ходе исследовании вселенной (High Time Resolution Universe Survey) с целью определения одного из типа нейтронных звезд — пульсаров. Всего 17898 наблюдений, среди них 1639 — пульсары. 

По каждому наблюдению доступно 8 непрерывных предикторов: среднее значение интегрального профиля; стандартное отклонение интегрального профиля; избыточный эксцесс интегрального профиля; асимметрия интегрального профиля; среднее значение кривой DM-SNR; стандартное отклонение кривой DM-SNR; избыточный эксцесс кривой DM-SNR; асимметрия кривой DM-SNR и бинарный отклик.
### Задание
![](https://github.com/InsightofSPb/Introduction_to_ML_and_Advanced_ML/blob/main/Tasks/Pulsars.png?raw=true)
Также необходимо указать число строк в полученной выборке и определить выборочное среднее для столбца MIP
```python
import pandas as pd

ap = pd.read_csv('Pulsar.csv')
p1 = ap[((ap.TG == 0) & (94.6640625 <= ap.MIP) & (ap.TG == 0) & (ap.MIP <= 95.2890625)) | ((ap.TG == 1) & (65.078125 <= ap.MIP) & (ap.TG == 1) & (ap.MIP <= 70.7421875))]

w = p1.MIP.mean()
print(p1.shape) # 202 строки
print(w) # 81.34
```
## Задание 2
Таблица ROSSTAT_SALARY_RU, содержит сведения о средней заработной плате в РФ по регионам на 1 января 2019 года по данным Росстата. Представим ситуацию, что из-за невнимательности операциониста, регионы: Республика Мордовия, Курская область оказались не представлены в итоговой сводке. Роль невнимательного операциониста придется исполнить Вам (т.е. нужно удалить соответствующие строки), а далее работать уже с новой выборкой.\
Постройте вариационный ряд, найти выборочное среднее, определить выборочную медиану.
```python
zp = pd.read_csv('Zp.csv', delimiter=',', index_col='REGION_NAME')
zp = zp.drop(['Курская область'])
zp = zp.drop(['Республика Мордовия'])
zp = zp.sort_values('SALARY')
sr = zp.SALARY.mean()
med = zp.SALARY.median()
print(zp)
print('X17:', zp.SALARY[16], 'X40:',zp.SALARY[39], 'X41:', zp.SALARY[40]) # в задании необходимо было вывести определённые значения из выборки
print('Среднее:', round(sr,2), 'Медиана:', med, sep='\n')
```