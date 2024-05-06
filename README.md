# Regresi-Linier-Sederhana
Memodelkan data dengan metode Regresi Linier Sederhana. Menggunakan library pandas, scipy, dan numpy. Kode ini terdiri dari pemanggilan data, data pre-processing, data visualisasi, pemodelan data, dan hasil analisis
# Memanggil dataset 
import pandas as pd 
df = pd.read_csv("DataStatreg.csv", sep=';') 
df
df.head()
# Hilangkan kolom pertama dan mengganti nama variabel
df.drop('Data ke-', axis=1, inplace=True) 
df.rename(columns={'Biaya iklan ($ thousand)':'x','Jumlah produk terjual (million)':'y'}, inplace=True)
df.head()
# Library untuk memunculkan Plot
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('fivethirtyeight') 
import warnings 
warnings.filterwarnings('ignore') 
%matplotlib inline

# Untuk memunculkan Scatter Plot
plt.scatter(df['x'], df['y']) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('Scatter Plot Biaya iklan ($ thousand) vs Jumlah produk terjual (million)') 
plt.show()
#Mengitung nilai korelasi pearson
from scipy.stats import pearsonr 
# Convert dataframe into series
list1 = df['x'] 
list2 = df['y'] 
corr, _ = pearsonr(list1, list2) 
print('Koefisien Pearson: %.5f' % corr)
#Memodelkan dengan Regresi Linier Sederhana
import numpy as np 
import statsmodels.api as sm 
x = df[['x']] 
y = df['y'] 
x = sm.add_constant(x) 
model = sm.OLS(y, x).fit() 
print_model = model.summary() 
print(print_model)
prediksi = model.predict(x) 
print(prediksi.head())
residual=model.resid 
print(residual.head())
