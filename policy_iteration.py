import numpy as np


def reward(policy):
    # 1. do action follow policy
    first_loc = np.repeat(np.arange(0, 21)[:,np.newaxis], 20, axis=1)
    second_loc = np.repeat(np.arange(0, 21)[np.newaxis,:], 20, axis=0)
    print(first_loc)
    print(second_loc)
    exit()

    next_1st_loc = first_loc - policy
    next_1st_loc[next_1st_loc < 0] = 0
    next_1st_loc[next_1st_loc > 20] = 20

    next_2nd_loc = second_loc + policy
    next_2nd_loc[next_2nd_loc < 0] = 0
    next_2nd_loc[next_2nd_loc > 20] = 20

    # 2. compute request, return in 1st, 2nd location
    req_1st_loc = 3
    ret_1st_loc = 3
    req_2nd_loc = 4
    ret_2nd_loc = 2

    # 3. compute reward each states
    tmp_1st_loc = next_1st_loc - req_1st_loc + ret_1st_loc
    req_1st_loc = np.where(tmp_1st_loc > 0, (req_1st_loc), req_1st_loc + tmp_1st_loc)

    tmp_1st_loc[tmp_1st_loc < 0] = 0
    tmp_1st_loc[tmp_1st_loc > 20] = 20

    tmp_2nd_loc = next_2nd_loc - req_2nd_loc + ret_2nd_loc
    req_2nd_loc = np.where(tmp_2nd_loc > 0, (req_2nd_loc), req_2nd_loc + tmp_2nd_loc )

    tmp_2nd_loc[tmp_2nd_loc < 0] = 0
    tmp_2nd_loc[tmp_2nd_loc > 20] = 20
    
    r_1st_loc = 10*req_1st_loc
    r_2nd_loc = 10*req_2nd_loc
    r = r_1st_loc + r_2nd_loc

    return r, tmp_1st_loc, tmp_2nd_loc

def policy_evaluation(policy, value_func, gamma=0.9, eps=10e-3):
    prev_value_func = np.copy(value_func)
    i=0
    v = np.zeros((20,20))
    while(np.max(np.abs(value_func-prev_value_func))>eps or i==0):
        prev_value_func = value_func
        r, next_1st_loc, next_2nd_loc = reward(policy)
        for i in range (20):
            for j in range (20):
                v[i][j] = prev_value_func[int(next_1st_loc[i][j])-1][int(next_2nd_loc[i][j])-1]
        value_func = (r + gamma*v)
        print(value_func)
    return value_func

def compute_Q_value(policy, value_func, gamma=0.9):
    r, next_1st_loc, next_2nd_loc = reward(policy)
    v = np.zeros((20,20))
    for i in range (20):
        for j in range (20):
            v[i][j] = value_func[next_1st_loc[i][j]-1][next_2nd_loc[i][j]-1]
    Q_value = (r + gamma*v)
    return Q_value

def policy_improvement(policy, value_func):
    # 1. compute Q value for all action (20x20x#actions)
    Q_value = np.zeros((20,20,11))
    Q_value[:,:,0] = compute_Q_value(np.full((20,20), -5), value_func)
    Q_value[:,:,1] = compute_Q_value(np.full((20,20), -4), value_func)
    Q_value[:,:,2] = compute_Q_value(np.full((20,20), -3), value_func)
    Q_value[:,:,3] = compute_Q_value(np.full((20,20), -2), value_func)
    Q_value[:,:,4] = compute_Q_value(np.full((20,20), -1), value_func)
    Q_value[:,:,5] = compute_Q_value(np.full((20,20), 0), value_func)
    Q_value[:,:,6] = compute_Q_value(np.full((20,20), 1), value_func)
    Q_value[:,:,7] = compute_Q_value(np.full((20,20), 2), value_func)
    Q_value[:,:,8] = compute_Q_value(np.full((20,20), 3), value_func)
    Q_value[:,:,9] = compute_Q_value(np.full((20,20), 4), value_func)
    Q_value[:,:,10] = compute_Q_value(np.full((20,20), 5), value_func)

    # 2. argmax값을 취하면 policy
    policy = Q_value.argmax(axis=2)
    return policy

def policy_iteration():
    eps = 10e-3
    prev_policy = np.ones((20,20))
    policy = np.zeros((20,20))
    # policy = np.random.randint(-5,5,size=(20,20))
    value_func = np.zeros((20,20))
    
    while(np.max(np.abs(policy-prev_policy))>eps):
        print('policy evaluation...')
        value_function = policy_evaluation(policy, value_func)
        
        print('policy improvement...')
        prev_policy = policy
        policy = policy_improvement(policy, value_function)
        print(policy)
    
    return 0

if __name__ == "__main__":
    policy_iteration()