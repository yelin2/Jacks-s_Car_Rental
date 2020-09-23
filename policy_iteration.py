import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()

state_dict = {}
nS = 441

for i in range(nS):
    state_dict[i] = (int(i/21), int(i%21))

def policy_evaluation(nS, nA, P, policy, value_func, gamma=0.9, eps=10e-3):
    while(1):
        new_value_func = np.copy(value_func)
        for s in range(nS):
            tmp = 0
            # transition is deterministic
            next_state, reward = P[s][policy[s]+5]
            tmp = value_func[next_state]
            new_value_func[s] = reward + gamma*tmp
        value_func = new_value_func
        if(np.max(np.abs(value_func-new_value_func))<eps):
            break
    return value_func

def policy_improvement(nS, nA, P, policy, value_func, gamma=0.9):
    Q_value = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            next_state, reward = P[s][a]
            tmp = value_func[next_state]
            Q_value[s][a] = reward + gamma*tmp
    new_policy = np.argmax(Q_value, axis=1) - 5
    print(policy.reshape(21,21))
    return new_policy

# TODO: 여기 아마 잘못된듯
def compute_transition(nS, nA):
    P = [[None for _ in range(nA)] for _ in range(nS)]
    req_1st = 3
    ret_1st = 3
    req_2nd = 4
    ret_2nd = 2

    for s in range(nS):
        for a in range(nA):
            a = a-5
            # 1st location to 2nd location
            if a > 0:
                # action a is available
                if state_dict[s][0] - a > 0 and state_dict[s][1] + a <20:
                    # state after taking action
                    tmp_1st_s = state_dict[s][0] - a
                    tmp_2nd_s = state_dict[s][1] + a
                    move = a
                    if tmp_1st_s - req_1st + ret_1st > 0:
                        r_1st = 10*req_1st
                        next_1st_s = tmp_1st_s - req_1st + ret_1st if tmp_1st_s - req_1st + ret_1st < 20 else 20
                    else:
                        r_1st = 10*(tmp_1st_s+ret_1st)
                        next_1st_s = 0
                    if tmp_2nd_s - req_2nd > 0:
                        r_2nd = 10*req_2nd
                        next_2nd_s = tmp_2nd_s - req_2nd + ret_2nd if tmp_2nd_s - req_2nd + ret_2nd < 20 else 20
                    else:
                        r_2nd = 10*tmp_2nd_s
                        # next_2nd_s = ret_2nd if ret_2nd < 20 else 20
                        next_2nd_s = 0
                    
                else:
                    move = min(state_dict[s][0], 20-state_dict[s][1])
                    tmp_1st_s = state_dict[s][0] - move
                    tmp_2nd_s = state_dict[s][1] + move
                    if tmp_1st_s - req_1st > 0:
                        r_1st = 10*req_1st
                        next_1st_s = tmp_1st_s - req_1st + ret_1st if tmp_1st_s - req_1st + ret_1st < 20 else 20
                    else:
                        r_1st = 10*tmp_1st_s
                        next_1st_s = ret_1st if ret_1st < 20 else 20
                    if tmp_2nd_s - req_2nd > 0:
                        r_2nd = 10*req_2nd
                        next_2nd_s = tmp_2nd_s - req_2nd + ret_2nd if tmp_2nd_s - req_2nd + ret_2nd < 20 else 20
                    else:
                        r_2nd = 10*tmp_2nd_s
                        next_2nd_s = ret_2nd if ret_2nd < 20 else 20
            else:
                if state_dict[s][0] - a < 20 and state_dict[s][1] + a > 0:
                    tmp_1st_s = state_dict[s][0] - a
                    tmp_2nd_s = state_dict[s][1] + a
                    move = a
                    if tmp_1st_s - req_1st > 0:
                        r_1st = 10*req_1st
                        next_1st_s = tmp_1st_s - req_1st + ret_1st if tmp_1st_s - req_1st + ret_1st < 20 else 20 
                    else:
                        r_1st = 10*tmp_1st_s
                        next_1st_s = ret_1st if ret_1st <20 else 20
                    if tmp_2nd_s - req_2nd > 0:
                        r_2nd = 10*req_2nd
                        next_2nd_s = tmp_2nd_s - req_2nd + ret_2nd if tmp_2nd_s - req_2nd + ret_2nd < 20 else 20
                    else:
                        r_2nd = 10*tmp_2nd_s
                        next_2nd_s = ret_2nd if ret_2nd < 20 else 20
                else:
                    move = min(state_dict[s][1], 20-state_dict[s][0])
                    tmp_1st_s = state_dict[s][0] - move
                    tmp_2nd_s = state_dict[s][1] + move 
                    if tmp_1st_s - req_1st > 0:
                        r_1st = 10*req_1st
                        next_1st_s = tmp_1st_s - req_1st + ret_1st if tmp_1st_s - req_1st + ret_1st < 20 else 20
                    else:
                        r_1st = 10*tmp_1st_s
                        next_1st_s = ret_1st if ret_1st < 20 else 20
                    if tmp_2nd_s - req_2nd > 0:
                        r_2nd = 10*req_2nd
                        next_2nd_s = tmp_2nd_s - req_2nd + ret_2nd if tmp_2nd_s - req_2nd + ret_2nd < 20 else 20
                    else:
                        r_2nd = 10*tmp_2nd_s
                        next_2nd_s = ret_2nd if ret_2nd < 20 else 20                                   
            print(next_1st_s, next_2nd_s)
            exit()
            tmp = [s for s, loc in state_dict.items() if loc == (next_1st_s, next_2nd_s)]
            P[s][a+5] = (tmp[0], r_1st+r_2nd)
    print(P)
    exit()

    return P

def policy_iteration():
    eps = 10e-3
    nS = 441
    nA = 11
    
    prev_policy = np.ones(nS, dtype=int)
    policy = np.random.randint(-5, 5, nS, dtype=int)
    value_function = np.zeros(nS)
    
    P = compute_transition(nS, nA)
    i = 0
    while(np.sum(np.abs(policy-prev_policy))>eps):
        print('policy evaluation...')
        value_function = policy_evaluation(nS, nA, P, policy, value_function)
        print(value_function)

        print('policy improvement...')
        prev_policy = policy
        policy = policy_improvement(nS, nA, P, policy, value_function)
        # print(policy.reshape((21,21)))

    return value_function, policy

def print_values(v, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Values ')
    aZ = []
    aX = []
    aY = []
    for i in range (20):
        for j in range (20):
            aX.append(i)
            aY.append(j)
            tmp = [s for s, loc in state_dict.items() if loc == (i, j)]
            aZ.append(v[tmp])
    ax.set_ylabel('# of cars at location 1')
    ax.set_xlabel('# of cars at location 2')
    ax.scatter(aX, aY, aZ)
    # plt.show()
    plt.savefig('value{}.png'.format(i))

def visualize_policy_plot(pi, title, fig_file):
    pi_arr = np.zeros((21,21))
    for k, v in state_dict.items():
        pi_arr[v[0]][v[1]] = pi[k]
    pi_arr = pi.reshape((21,21))

    f, _ = plt.subplots()
    plt.pcolormesh(pi_arr)
    plt.colorbar()

    plt.title(title)
    plt.ylabel('Cars in Location 1')
    plt.xlabel('Cars in Location 2')

    f.savefig(fig_file)

if __name__ == "__main__":
    v, p = policy_iteration()
    print_values(v, 1)
    visualize_policy_plot(p, 'policy', 'policy3.png')