import xlrd
import numpy as np
data = xlrd.open_workbook("rain_case_2018.xlsx")
table = data.sheets()[0]

rain_case_0 = []
for i in range(288):
    rain_case_0.append(0)

for i in range(37):
    precipVal = table.cell(i+21875,1).value
    rain_case_0.append(precipVal)
    rain_case_0.append(precipVal)
    rain_case_0.append(precipVal)

for i in range(399, 288*3):
    rain_case_0.append(0)

rain_case_1 = []
for i in range(288):
    rain_case_1.append(0)
for i in range(17):
    precipVal = table.cell(i+29853,1).value
    rain_case_1.append(precipVal)
    rain_case_1.append(precipVal)
    rain_case_1.append(precipVal)
for i in range(339, 288*3):
    rain_case_1.append(0)


rain_case_2 = []
for i in range(288):
    rain_case_2.append(0)
for i in range(16):
    precipVal = table.cell(i+32699,1).value
    rain_case_2.append(precipVal)
    rain_case_2.append(precipVal)
    rain_case_2.append(precipVal)
for i in range(336, 288*3):
    rain_case_2.append(0)

rain_case_3 = []
for i in range(288):
    rain_case_3.append(0)
for i in range(22):
    precipVal = table.cell(i+32699,1).value
    rain_case_3.append(precipVal)
    rain_case_3.append(precipVal)
    rain_case_3.append(precipVal)
for i in range(354, 288*3):
    rain_case_3.append(0)


def rain_generation(type_num):

    rain = []

    for i in range(288):
        rain.append(0)
    for i in range(288,300):
        rain.append(0.1)
    for i in range(300,312):
        rain.append(0.2)
    for i in range(312,336):
        rain.append(0.4)
    for i in range(336,384):
        rain.append(0.5)
    for i in range(384,408):
        rain.append(0.4)
    for i in range(408,420):
        rain.append(0.2)
    for i in range(420,432):
        rain.append(0.1)
    for i in range(432,288*3):
        rain.append(0)

    rain1=rain

    rain = []

    for i in range(288):
        rain.append(0)
    for i in range(288,300):
        rain.append(0.5)
    for i in range(300,312):
        rain.append(0.4)
    for i in range(312,336):
        rain.append(0.2)
    for i in range(336,384):
        rain.append(0.1)
    for i in range(384,408):
        rain.append(0.2)
    for i in range(408,420):
        rain.append(0.4)
    for i in range(420,432):
        rain.append(0.5)
    for i in range(432,288*3):
        rain.append(0)

    rain2=rain

    rain = []

    for i in range(288):
        rain.append(0)
    for i in range(288,300):
        rain.append(0.3)
    for i in range(300,312):
        rain.append(0.3)
    for i in range(312,336):
        rain.append(0.3)
    for i in range(336,384):
        rain.append(0.3)
    for i in range(384,408):
        rain.append(0.3)
    for i in range(408,420):
        rain.append(0.3)
    for i in range(420,432):
        rain.append(0.3)
    for i in range(432,288*3):
        rain.append(0)

    rain3=rain

    if type_num==1:
        return rain1
    if type_num==2:
        return rain2
    if type_num==3:
        return rain3

rain_1=rain_generation(1)
rain_2=rain_generation(2)
rain_3=rain_generation(3)
print(rain_case_1)
#print(rain_case_2)
#print(rain_case_3)
print(rain_1)
#print(rain_2)
#print(rain_3)
print(len(rain_case_1))
print(len(rain_case_2))
print(len(rain_case_3))
print(len(rain_1))
print(len(rain_2))


action_list=[[] for i in range(20)]
print(action_list)

episode_count = 500 # 5000
timesteps = episode_count*576

epi_start =1
epi_end=0
epsilon_value=[]
epsilon_value = np.linspace(epi_start, epi_end, timesteps)
print(epsilon_value)
