# EXNO-5-DS-DATA VISUALIZATION USING MATPLOT LIBRARY

# Aim:
  To Perform Data Visualization using matplot python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
 
# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

x = [2018, 2019, 2020, 2021, 2022]

y = [15000, 20000, 25000, 30000, 35000]

plt.plot(x, y, 'g*', linestyle='dashed', linewidth=2, markersize=12)

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.title('2D Diagram')

plt.show()

<img width="652" height="455" alt="image" src="https://github.com/user-attachments/assets/089eb41c-ec5f-465e-a485-9f30196c0f3f" />

plt.subplot(2,2,1)

plt.plot(x,y,'--')

plt.subplot(2,2,2)

plt.plot(x,y,'o--')

plt.subplot(2,2,3)

plt.plot(x,y,'bo')

plt.subplot(2,2,4)

plt.plot(x,y,'go')

plt.show()

<img width="745" height="502" alt="image" src="https://github.com/user-attachments/assets/3fdcf838-108d-4d46-b83e-0e1c4e0f29ad" />

x = np.arange(0, 4*np.pi, 0.1)

y = np.sin(x)

plt.title("sine waveform")

plt.plot(x,y)

plt.show()

<img width="666" height="460" alt="image" src="https://github.com/user-attachments/assets/a9fe9b0b-8098-4612-9686-351f8959f1e1" />

x = [2, 8, 10]

y = [11, 16, 9]

x1 = [3, 4, 11] 

y1 = [6, 15, 7]

plt.bar(x, y, color='r')

plt.bar(x1, y1, color='g')

plt.title("Bar graph")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.show()

<img width="712" height="479" alt="image" src="https://github.com/user-attachments/assets/f77115fc-d585-486c-b50e-0738fb6fb9f8" />

x = [1, 2, 3]

y = [7, 3, 9]

plt.plot(x, y, color='g')

plt.title("line graph")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.show()

<img width="789" height="537" alt="image" src="https://github.com/user-attachments/assets/82c9074a-72ae-44f8-865e-1a147da1f9ac" />

x1 = [1, 2, 3]

y1 = [2, 4, 1]

plt.plot(x1, y1, label='line1')

x2 = [1, 2, 3]  

y2 = [4, 1, 3]

plt.plot(x2, y2, label='line2')

plt.title("multiline graph")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.legend()

plt.show()

<img width="665" height="483" alt="image" src="https://github.com/user-attachments/assets/11b5faf2-f27f-4d44-8bc3-7244e013c86e" />

x = [1, 2, 3, 4, 5, 6]

y = [2, 4, 1, 5, 2, 6]

plt.plot(x, y, color='green', linestyle='dashed', linewidth=3, 
         marker='o', markerfacecolor='red', markersize=12, label='spices')
         
plt.xlim(1, 8)

plt.ylim(1, 8)

plt.title("line graph")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.legend()

plt.show()

<img width="682" height="501" alt="image" src="https://github.com/user-attachments/assets/cfc8d57c-5d74-4bfe-9d64-9d36fee931c1" />

yield_apples=[0.895,0.91,0.919,0.926,0.931]

plt.plot(yield_apples,linestyle='dashed',linewidth=3,marker='*',markersize=12,color='g')

plt.show()

<img width="626" height="430" alt="image" src="https://github.com/user-attachments/assets/d1bea539-38b5-4c00-bc11-3bfbccc1d425" />

oranges = [2, 4, 6, 7, 8, 12, 45]

apples = [2, 4, 5, 6, 8, 23, 37]

years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

plt.plot(years, apples, marker='o')

plt.plot(years, oranges, marker='*')

plt.title('Crop yields in Kanto')

plt.xlabel('Year')

plt.ylabel('Yield (tons per hectare)')

plt.legend(['apples', 'oranges'])  

plt.show()

<img width="737" height="559" alt="image" src="https://github.com/user-attachments/assets/c0668664-1852-4e3a-85de-74edd15a809e" />

oranges = [2, 4, 6, 7, 8, 12]

apples = [2, 4, 5, 6, 8, 23]

years = [2019, 2020, 2021, 2022, 2023, 2024]

plt.bar(oranges,apples)

plt.plot(years, apples, label='apples', marker='*')

plt.plot(years, oranges, label='oranges', marker='o')

plt.title('Fruit sales')  

plt.xlabel('Year')  

plt.ylabel('Sales')  

plt.legend()  

plt.show()

<img width="758" height="528" alt="image" src="https://github.com/user-attachments/assets/310a7a75-bbb3-4a89-9935-0669d6dd5809" />

plt.figure(figsize=(12,6))

plt.plot(years, oranges, marker='o',label='oranges')

plt.title("yield of oranges (tons per hectare)")

plt.legend()

<img width="813" height="437" alt="image" src="https://github.com/user-attachments/assets/ddbdaa90-38f5-4fe9-a82c-fc5e5d5086e6" />

import matplotlib.pyplot as plt

print("scatter plot is:")

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]

plt.scatter(x, y, label='star', color='green', marker='*', s=30)

plt.title("my scatterplot!")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.legend()

plt.show()

<img width="689" height="498" alt="image" src="https://github.com/user-attachments/assets/9f99b597-5ad1-4d7b-9b79-d9b97b1a248f" />

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]

y1 = [10, 12, 14, 16, 18]

y2 = [5, 7, 9, 11, 13]

y3 = [2, 4, 6, 8, 10]

plt.fill_between(x, y1, color='green',label='y1')

plt.fill_between(x, y2, color='blue', label='y2')

plt.fill_between(x, y3, color='red',label='y3')

plt.title("Fill Between Example")

plt.legend()

plt.show()

<img width="692" height="539" alt="image" src="https://github.com/user-attachments/assets/414ae6a3-1402-4ab6-a588-38d677197008" />

plt.stackplot(x,y1,y2,y3,labels=['line1','line2','line3'])

plt.legend(loc='upper left')

plt.title("Stacked line chart")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.show()

<img width="643" height="514" alt="image" src="https://github.com/user-attachments/assets/459e7f38-29ec-46a2-964c-b7712b090dcb" />

from scipy.interpolate import make_interp_spline

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

y = np.array([2, 4, 5, 7, 8, 9, 10, 11, 12, 13])

spl = make_interp_spline(x, y)

x_smooth = np.linspace(x.min(), x.max(), 100)

y_smooth = spl(x_smooth)

plt.plot(x, y, 'o', label='data')

plt.plot(x_smooth, y_smooth, '-', label='spline')

plt.legend()

plt.show()

<img width="726" height="557" alt="image" src="https://github.com/user-attachments/assets/66da54a6-09ca-4396-bfc0-2c7b28802ca1" />

values=[5,6,7,3,2]

names=['A','B','C','D','E']

plt.bar(names,values,color='green')

plt.show()

<img width="696" height="471" alt="image" src="https://github.com/user-attachments/assets/daf2ce63-d639-43ac-a66c-adc40d2dbe4a" />

ages = [2, 5, 70, 40, 45, 50, 43, 40, 44, 60, 7, 13, 57, 18, 90, 77, 32, 21, 20, 40]

range1 = (0, 100)

bins = 10

plt.hist(ages, bins, range1, color='green', histtype='bar', rwidth=0.8)

plt.xlabel('ages')

plt.ylabel('no. of people')

plt.title('my histogram')

plt.show()

<img width="645" height="552" alt="image" src="https://github.com/user-attachments/assets/e9e50dcc-aa6e-4c68-bd74-52db454aab4f" />

x=[2,1,6,2,4,8,9,4,2,4,10,6,4,5,7,7,3,2,7,5,3,5,9,2,1]

plt.hist(x,bins=10,color='green',alpha=0.5)

plt.show()

<img width="732" height="542" alt="image" src="https://github.com/user-attachments/assets/d565991c-afa7-43ad-ba3d-956e14b93a12" />

np.random.seed(0)

data=np.random.normal(loc=0,scale=1,size=100)

data

<img width="669" height="426" alt="image" src="https://github.com/user-attachments/assets/b99b5c87-0bb1-4657-96c8-3a160795b114" />

fig, ax = plt.subplots()

ax.boxplot(data)

ax.set_xlabel('x-axis')

ax.set_ylabel('y-axis')

ax.set_title('box plot')

<img width="734" height="580" alt="image" src="https://github.com/user-attachments/assets/29ff3655-31b4-4d37-817c-ef9a951f6749" />

activities=['eat', 'sleep', 'work', 'play']


slices=[3,7,8,6]

colors=['y', 'g', 'b', 'r']

plt.pie(slices, labels=activities, colors=colors, startangle=90, shadow=True, explode=(0,0,0.1,0), radius=1.2, autopct="X1.1")

plt.legend()

plt.show()

<img width="469" height="376" alt="image" src="https://github.com/user-attachments/assets/89bfc2c4-751d-4847-bee3-bb8383143392" />

labels='python', 'C+', 'ruby', 'java'

sizes=[215,130,245,210]

colors=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode=(0,0.4,0,0.5)

plt.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()

<img width="536" height="433" alt="image" src="https://github.com/user-attachments/assets/0898ef09-7fcf-4e22-85c4-36743933d468" />







# Result:
Thus,all the data visualization techniques of matploylib has been implemented
