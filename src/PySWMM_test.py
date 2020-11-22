# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:39:19 2019

@author: xzw
"""

import pyswmm
from pyswmm.swmm5 import PySWMM




swmm_model = PySWMM('test1.inp','test1.rpt','test1.out')
swmm_model.swmm_open()
swmm_model.swmm_start()
a=[{} for i in range(10000)]
step=0
while(True):

#    time = swmm_model.swmm_step()
    flooding_before_step = swmm_model.flow_routing_stats()['flooding']

    swmm_model.setLinkSetting('Pump_ChengXi1',1)

    time = swmm_model.swmm_stride(300)


    #print(time)
    step +=1
    if (time <= 0.0): break

swmm_model.swmm_end()
swmm_model.swmm_report()
swmm_model.swmm_close()
