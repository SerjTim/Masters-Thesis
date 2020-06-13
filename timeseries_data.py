import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

radovic2 = pd.read_csv('wpd_datasets2.csv')
# print (radovic2.columns)

radovic2.columns = ['day_x','day_y', 'night_x','night_y','sterowanie_x','sterowanie_y']
radovic2.drop([0], inplace= True)


# # we're not losing much by substituting x
# a = range(1,25)
# day_error = np.array(list(a)) - radovic2['sterowanie_x'].apply(float)
# sns.distplot(day_error)

radovic = radovic2.drop(['day_x','night_x','sterowanie_x'], axis=1)
print (radovic.columns)
radovic = radovic.applymap(float)
radovic.plot()
plt.title('Porównanie strategii ładowania')
plt.legend(['dzienna','nocna','sterowalna'])
plt.xlabel('godz.')
plt.ylabel('kW/pojazd')
plt.savefig('strategie ladowania')
plt.show()