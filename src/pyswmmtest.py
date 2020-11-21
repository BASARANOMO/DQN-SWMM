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
from pyswmm import Simulation, Links, Nodes, SystemStats
from pyswmm.swmm5 import PySWMM
import matplotlib.pyplot as plt


#import from previous work
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import reward_function, epsi_greedy, swmm_track
from ger_fun import build_network, swmm_states
from core_network import stacker, replay_stacker







    


#def main():
    
#    load_model = (sys.argv[3])
#    write_model = (sys.argv[4])
#    epi_start = float(sys.argv[1])
#    epi_end = float(sys.argv[2])
        
    #vaiables define


    
    #data prepare
action = np.zeros((288*2, 7), dtype=np.int)
rain = np.zeros((288*3, 1), dtype=np.float)
for i in range(288*2):
    for j in range(7):
        action[i][j]=random.randint(0,1)
for i in range(288*3):
    rain[i]=random.random()
action_space = np.linspace(0, 4, 5)

#initial memory
node_controlled=['ChengXi']
controlled_ponds = {}
for i in node_controlled:
    controlled_ponds[i] = pond_tracker(i,i,1,576)

#build neural model
models_ac = {}
for i in node_controlled:
    model = target = build_network(1,5,2, 50, 'relu', 0.0)
    #def build_network(input_states,output_states,hidden_layers,nuron_count,activation_function,dropout):
#        if load_model != 'initial_run':
#            model.load_weights(i + load_model)

    target.set_weights(model.get_weights())
    models_ac[i] = [model, target]


# Initialize Deep Q agents
agents_dqn = {}
for i in node_controlled:
    temp = deep_q_agent(models_ac[i][0],
                    models_ac[i][1],
                    1,
                    controlled_ponds[i].replay_memory,
                    epsi_greedy)
    agents_dqn[i] = temp

# Simulation Time Steps

episode_count = 198
timesteps = episode_count*576

epi_start =0
epi_end=1
epsilon_value = np.linspace(epi_start, epi_end, timesteps + 10)

episode_timer=0

while episode_timer<episode_count:
    
    
    #run SWMM
    swmm_model = PySWMM('test.inp','test.rpt','test.out')
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
    print ('Episode_epoch: ', episode_timer)
    

    step=0    
        
        
    while(True):
        step +=1
        # Take a look at whats happening
        for i in node_controlled:
            agents_dqn[i].state_vector = [swmm_model.getNodeResult(node[1],5)]
        # Take action
        for i in node_controlled:
            action_step = agents_dqn[i].actions_q(epsilon_value[step],action_space)
            agents_dqn[i].action_vector = action_step
            if action_step==0:
                swmm_model.setLinkSetting(pump[3],0)
                swmm_model.setLinkSetting(pump[4],0)
                swmm_model.setLinkSetting(pump[5],0)
                swmm_model.setLinkSetting(pump[6],0)
            if action_step==1:
                swmm_model.setLinkSetting(pump[3],1)
                swmm_model.setLinkSetting(pump[4],0)
                swmm_model.setLinkSetting(pump[5],0)
                swmm_model.setLinkSetting(pump[6],0)                   
            if action_step==2:
                swmm_model.setLinkSetting(pump[3],1)
                swmm_model.setLinkSetting(pump[4],0)
                swmm_model.setLinkSetting(pump[5],0)
                swmm_model.setLinkSetting(pump[6],0) 
            if action_step==3:
                swmm_model.setLinkSetting(pump[3],1)
                swmm_model.setLinkSetting(pump[4],1)
                swmm_model.setLinkSetting(pump[5],1)
                swmm_model.setLinkSetting(pump[6],0) 
            if action_step==4:
                swmm_model.setLinkSetting(pump[3],1)
                swmm_model.setLinkSetting(pump[4],1)
                swmm_model.setLinkSetting(pump[5],1)
                swmm_model.setLinkSetting(pump[6],1) 
                
            current_gate = agents_dqn[i].action_vector
            
        flooding_before_step = swmm_model.flow_routing_stats()['flooding'] 
#        print(swmm_model.getCurrentSimulationTime())
        
        #run swmm step
        time = swmm_model.swmm_stride(300)
    
    
        # reward function
        flooding_after_step = swmm_model.flow_routing_stats()['flooding'] 
        for i in node_controlled:
            agents_dqn[i].rewards_vector = [-10*(flooding_after_step- flooding_before_step)]
    
        # Observe the new states
        for i in node_controlled:
            agents_dqn[i].state_new_vector = [swmm_model.getNodeResult(node[1],5)]
            
    
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
            temp = [swmm_model.getNodeResult(node[1],5),swmm_model.getLinkResult(pump[3],0),swmm_model.getLinkResult(pump[3],0),swmm_model.getLinkResult(pump[3],0)]
    
            temp = np.append(temp, np.asarray([agents_dqn[i].action_vector, agents_dqn[i].rewards_vector]))
            controlled_ponds[i].tracker_update(temp)            
        # Train
        if step % 100 == 0:
            for i in node_controlled:
                agents_dqn[i].train_q(step)
    
            
            
        if (time <= 0.0): break
    
    swmm_model.swmm_end()
    swmm_model.swmm_report()
    swmm_model.swmm_close()
        
    
    for i in node_controlled:
        controlled_ponds[i].record_mean()
        print(controlled_ponds[i].bookkeeping['mean_rewards']._data[episode_timer-1])




         



#if __name__ == '__main__':
#    main()
#plot
#plt.plot(action[500:550,0])
#plt.plot(state_pump[500:550])