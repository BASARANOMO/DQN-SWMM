"""
01032021
"""

# import modules
import numpy as np
import pandas as pd
import sys
import random
import datetime
import pyswmm
import xlrd

from pyswmm.swmm5 import PySWMM
import matplotlib.pyplot as plt

sys.path.append('..')

# import
from Distributional_DQN.pond_net import pond_tracker
from Distributional_DQN.C51Agent import C51Agent
from Distributional_DQN.ger_fun import reward_function, epsi_greedy
from Distributional_DQN.ger_fun import build_network
from Distributional_DQN.core_network import stacker, replay_stacker

# test case
def rain_generation(type_num):
    rain = []
    if type_num == 1:
        rain = rain + [0] * 288 + [0.1] * 12 + [0.2] * 12 + [0.4] * 24 + [0.5] * 48 + [0.4] * 24 + [0.2] * 12 + [0.1] * 12 + [0] * 432
    elif type_num == 2:
        rain = rain + [0] * 288 + [0.5] * 12 + [0.4] * 12 + [0.2] * 24 + [0.1] * 48 + [0.2] * 24 + [0.4] * 12 + [0.5] * 12 + [0] * 432
    elif type_num == 3:
        rain = rain + [0] * 288 + [0.3] * 12 + [0.3] * 12 + [0.3] * 24 + [0.3] * 48 + [0.3] * 24 + [0.3] * 12 + [0.3] * 12 + [0] * 432
    return rain


starttime=datetime.datetime.now()
#initial memory
state_controlled=9

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
    model = target = build_network(state_controlled,20,3, 10, 'relu', 0.0)
