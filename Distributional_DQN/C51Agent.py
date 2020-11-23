import numpy as np


def randombatch(sample_size, replay_size):
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx


class C51Agent:
    def __init__(self,
                 action_value_model,
                 target_model,
                 states,
                 replay_memory,
                 policy,
                 v_max = 10,
                 v_min = -10,
                 num_atoms = 51,
                 batch_size=128,
                 target_update=10000,
                 train=True):

        self.states = states
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.policy = policy
        self.train = train
        self.target_update = target_update

        # C51 parameters
        self.v_max = v_max
        self.v_min = v_min
        self.num_atoms = num_atoms
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.state_vector = np.zeros((1, self.states))
        self.state_new_vector = np.zeros((1, self.states))
        self.rewards_vector = np.zeros((1))
        self.terminal_vector = np.zeros((1))
        self.action_vector = np.zeros((1))

        self.training_batch = {'states': np.zeros((self.batch_size,
                                                   self.states)),
                               'states_new': np.zeros((self.batch_size,
                                                       self.states)),
                               'actions': np.zeros((self.batch_size, 1)),
                               'rewards': np.zeros((self.batch_size, 1)),
                               'terminal': np.zeros((self.batch_size, 1))}

    def _random_sample(self):
        indx = randombatch(self.batch_size, len(self.replay['states'].data()))
        for i in self.training_batch.keys():
            temp = self.replay[i].data()
            self.training_batch[i] = temp[indx]

    def _update_target_model(self):
        self.target_model.set_weights(self.ac_model.get_weights())

    def _train(self):
        temp_states_new = self.training_batch['states_new']
        temp_states = self.training_batch['states']
        temp_rewards = self.training_batch['rewards']
        temp_terminal = self.training_batch['terminal']
        temp_actions = self.training_batch['actions']

        z = self.ac_model.predict_on_batch(temp_states_new)
        z_ = self.target_model.predict_on_batch(temp_states_new)
        z_concat = np.vstack()
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, 20), order='F')
        next_actions = np.argmax(q, axis=1)

        m_prob = [np.zeros((self.batch_size, self.num_atoms)) for _ in range(20)]
        for i in range(self.batch_size):
            action_idx = int(temp_actions[i])
            if temp_terminal[i]:
                Tz = min(self.v_max, max(self.v_min, temp_reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[action_idx][i][int(l)] += (u - bj)
                m_prob[action_idx][i][int(u)] += (bj - l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, temp_reward[i] + 0.95 * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[action_idx][i][int(l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[action_idx][i][int(u)] += z_[next_actions[i]][i][j] * (bj - l)

        self.ac_model.fit(temp_states,
                          m_prob,
                          batch_size=128,
                          epochs=1, # change epoch for tests
                          verbose=0)

    def train_q(self, timesteps):
        self._random_sample()
        temp = True if timesteps > 100 else False
        if temp:
            self._update_target_model()
        self._train()

    def actions_q(self, action_space):
        z = self.ac_model.predict(self.state_vector)
        z_concat = np.vstack(z)
        q_values = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        action = self.policy(action_space, q_values, epsilon)
        return action

"""
    def actions_q(self, epsilon, action_space):
        q_values = self.ac_model.predict(self.state_vector)
        action = self.policy(action_space, q_values, epsilon)
        return action
"""
