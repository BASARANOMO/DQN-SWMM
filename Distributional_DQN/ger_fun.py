import numpy as np
#import matplotlib.pyplot as plt
#import swmm
from keras.models import Sequential, Model
from keras.layers import Input,Dense, Activation, Dropout
from keras.optimizers import RMSprop


# Reward Function
def reward_function(depth, outflow, gate_postions_rate, flood):
    depth = float(depth)
    outflow = float(outflow)
    gate_postions_rate = float(gate_postions_rate)
    reward_flow = 1.0 if outflow < 0.1 else -1.0
    reward_depth = -0.5*depth if depth < 2.0 else -depth**2 + 3.0
    reward_gate = 0.0 if gate_postions_rate < 0.4 else -10.0*gate_postions_rate
    reward_flood = -10.0*flood

    return reward_flow + reward_depth + reward_gate + reward_flood

# Policy Function
def epsi_greedy(action_space, q_values, epsilon):
    """Epsilon Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)

def build_network(input_states,
                  output_states,
                  hidden_layers,
                  nuron_count,
                  activation_function,
                  dropout):

    model = Sequential()
    model.add(Dense(nuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for _ in range(0, hidden_layers-1):
        model.add(Dense(nuron_count))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def build_network_C51(input_states,
                      output_states,
                      num_atoms,
                      hidden_layers,
                      nuron_count,
                      activation_function,
                      dropout):

    input_state = Input((input_states,))
    layer = input_state
    for _ in range(0, hidden_layers-1):
        layer = Dense(nuron_count, activation=activation_function)(layer)
        layer = Dropout(dropout)(layer)
    
    outputs = []
    for _ in range(output_states):
        outputs.append(Dense(num_atoms, activation='relu')(layer))
    
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    return Model(input_state, outputs, loss='mean_squared_error', optimizer=sgd)