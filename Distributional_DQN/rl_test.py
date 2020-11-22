# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:42:17 2019

@author: xzw
"""
import numpy as np
import pandas as pd
import sys
import random
import datetime
import pyswmm
import xlrd

from pyswmm.swmm5 import PySWMM
import matplotlib.pyplot as plt


#import from previous work
from pond_net import pond_tracker
from C51Agent import C51Agent
from ger_fun import reward_function, epsi_greedy
from ger_fun import build_network_C51
from core_network import stacker, replay_stacker

data = xlrd.open_workbook("rain_case_2018.xlsx")
table = data.sheets()[0]
###########################################################
# real rain case 0 : 21840 - 21941

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

###########################################################
# real rain case 1 : 29854
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

###########################################################
# real rain case 2 : 32700
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

###########################################################
# real rain case 3 : 32923
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

#def main():

#    load_model = (sys.argv[3])
#    write_model = (sys.argv[4])
#    epi_start = float(sys.argv[1])
#    epi_end = float(sys.argv[2])

    #vaiables define


starttime=datetime.datetime.now()
#initial memory
state_controlled=9

     #data prepare

action_space = np.linspace(0, 19, 20,dtype='int')
action_list=[[] for i in range(20)]
action_part1=[[] for i in range(4)]
action_part2=[[] for i in range(5)]
action_part1[0]=[0,0,0]
action_part1[1]=[1,0,0]
action_part1[2]=[1,1,0]
action_part1[3]=[1,1,1]
action_part2[0]=[0,0,0,0]
action_part2[1]=[1,0,0,0]
action_part2[2]=[1,1,0,0]
action_part2[3]=[1,1,1,0]
action_part2[4]=[1,1,1,1]
for i in range(4):
    for j in range(5):
        action_list[5*i+j]=action_part1[i]+action_part2[j]


#initial memory
node_controlled=['ChengXi']
controlled_ponds = {}
for i in node_controlled:
    controlled_ponds[i] = pond_tracker(i,i,state_controlled,1000000)

#build neural model
models_ac = {}
for i in node_controlled:
    model = target = build_network_C51(state_controlled, 20, 51, 3, 10, 'relu', 0.0)
    #def build_network(input_states,output_states,hidden_layers,nuron_count,activation_function,dropout):
#        if load_model != 'initial_run':
#            model.load_weights(i + load_model)

    target.set_weights(model.get_weights())
    models_ac[i] = [model, target]


# Initialize Deep Q agents
agents_dqn = {}
for i in node_controlled:
    temp = C51Agent(models_ac[i][0],
                    models_ac[i][1],
                    state_controlled,
                    controlled_ponds[i].replay_memory,
                    epsi_greedy)
    agents_dqn[i] = temp

# Simulation Time Steps

episode_count = 500 # 5000
timesteps = episode_count*576

epi_start =1
epi_end=0
epsilon_value=[]
epsilon_value = np.linspace(epi_start, epi_end, timesteps)

tempnum = np.zeros(5760)
epsilon_value=np.append(epsilon_value,tempnum)

print(len(epsilon_value))

episode_timer=0

while episode_timer<episode_count:

    i=random.randint(1,3)
    sim_num=i
    #rain = rain_case_0

    if i==1:
        rain=rain_case_1
    elif i==2:
        rain=rain_case_2
    else:
        rain=rain_case_3

    #run SWMM
    #rain=rain_generation(i)
    #sim_num=1
    #rain = rain_generation(1)

    #swmm_model = PySWMM('test_rain_case_0.inp','test_rain_case_0.rpt','test_rain_case_0.out')
    swmm_model = PySWMM('test_rain_case_'+str(i)+'.inp','test_rain_case_'+str(i)+'.rpt','test_rain_case_'+str(i)+'.out')
    #swmm_model = PySWMM('test'+str(i)+'.inp','test'+str(i)+'.rpt','test'+str(i)+'.out')
    swmm_model.swmm_open()
    swmm_model.swmm_start()
    n1="Tank_LaoDongLu"
    n2="Tank_ChengXi"
    p1="Pump_LaoDongLu1"
    p2="Pump_LaoDongLu2"
    p3="Pump_LaoDongLu3"
    p4="Pump_ChengXi1"
    p5="Pump_ChengXi2"
    p6="Pump_ChengXi3"
    p7="Pump_ChengXi4"
    node=[n1,n2]
    pump=[p1,p2,p3,p4,p5,p6,p7]

    for i in node_controlled:
        controlled_ponds[i].forget_past()
    episode_timer +=1
    print ('Episode_epoch: ', episode_timer,'Scenario: ',sim_num)


    step=0


    while(True):

        # Take a look at whats happening
        for i in node_controlled:
            temp=np.zeros((1,state_controlled))
            temp[0][0]=swmm_model.getNodeResult(node[0],5)
            temp[0][1]=swmm_model.getNodeResult(node[1],5)
            for j in range(2,8):
                temp[0][j]=rain[step+j-2]
            temp[0][8]=(step/12)%24
            agents_dqn[i].state_vector = temp

        for i in node_controlled:
            action_step = agents_dqn[i].actions_q(epsilon_value[episode_timer*576+step],action_space)
            agents_dqn[i].action_vector = action_step

        for j in range(7):
            swmm_model.setLinkSetting(pump[j],action_list[action_step][j])


            current_gate = agents_dqn[i].action_vector

        flooding_before_step = swmm_model.flow_routing_stats()['flooding']
        #print(swmm_model.getCurrentSimulationTime())

        #run swmm step
        time = swmm_model.swmm_stride(300)
        step +=1

        # reward function
        flooding_after_step = swmm_model.flow_routing_stats()['flooding']
        depth1=swmm_model.getNodeResult(node[0],5)
        depth2=swmm_model.getNodeResult(node[1],5)
        for i in node_controlled:
            reward=-10*(flooding_after_step- flooding_before_step)-30*(abs(depth1-3.4)+depth1-3.4)-100*(abs(depth2-4.7)+depth2-4.7)
            agents_dqn[i].rewards_vector = [reward]

        # Observe the new states
        for i in node_controlled:
            temp=np.zeros((1,state_controlled))
            temp[0][0]=swmm_model.getNodeResult(node[0],5)
            temp[0][1]=swmm_model.getNodeResult(node[1],5)
            for j in range(2,8):
                temp[0][j]=rain[step+j-2]
            temp[0][8]=(step/12)%24
            agents_dqn[i].state_new_vector = temp

        # Update Replay Memory
        for i in node_controlled:
            controlled_ponds[i].replay_memory_update(
                agents_dqn[i].state_vector,
                agents_dqn[i].state_new_vector,
                agents_dqn[i].rewards_vector,
                agents_dqn[i].action_vector,
                agents_dqn[i].terminal_vector)

        # Track Controlled ponds
        for i in node_controlled:
            temp = [swmm_model.getNodeResult(node[1],5),sim_num,swmm_model.getLinkResult(pump[3],0),flooding_after_step- flooding_before_step]

            temp = np.append(temp, np.asarray([agents_dqn[i].action_vector, reward]))
            controlled_ponds[i].tracker_update(temp)
        # Train
        if step % 100 == 0:
            for i in node_controlled:
                agents_dqn[i].train_q(episode_timer*576+step)

        if (time <= 0.0): break


    swmm_model.swmm_end()
    swmm_model.swmm_report()
    swmm_model.swmm_close()


    for i in node_controlled:
        controlled_ponds[i].record_mean()
        print(controlled_ponds[i].bookkeeping['mean_rewards']._data[episode_timer-1])



endtime=datetime.datetime.now()

print('elapsed time:',endtime-starttime)

a=controlled_ponds['ChengXi'].bookkeeping['mean_rewards'].data()
b=controlled_ponds['ChengXi'].bookkeeping['scenario_num'].data()

index=np.argwhere(b==1)
plt.plot(a[index[:,0]])
index=np.argwhere(b==2)
plt.plot(a[index[:,0]])
index=np.argwhere(b==3)
plt.plot(a[index[:,0]])
plt.show()
#if __name__ == '__main__':
#    main()
#plot
#plt.plot(action[500:550,0])
#plt.plot(state_pump[500:550])
