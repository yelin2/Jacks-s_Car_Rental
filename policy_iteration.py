import numpy as np

# def PolicyIteration()

def reward(policy):
    # 1. do action follow policy
    first_loc = np.repeat(np.arange(1, 21)[:,np.newaxis], 20, axis=1)
    second_loc = np.repeat(np.arange(1, 21)[np.newaxis,:], 20, axis=0)

    next_1st_loc = first_loc - policy
    next_1st_loc[next_1st_loc < 0] = 0
    next_1st_loc[next_1st_loc > 20] = 20

    next_2nd_loc = second_loc + policy
    next_2nd_loc[next_2nd_loc < 0] = 0
    next_2nd_loc[next_2nd_loc > 20] = 20

    # 2. compute request, return in 1st, 2nd location
    req_1st_loc = np.random.poisson(lam=3, size=(20,20))
    ret_1st_loc = np.random.poisson(lam=3, size=(20,20))
    req_2nd_loc = np.random.poisson(lam=4, size=(20,20))
    ret_2nd_loc = np.random.poisson(lam=2, size=(20,20))

    # 3. compute reward each states
    tmp_1st_loc = next_1st_loc - req_1st_loc + ret_1st_loc
    req_1st_loc = np.where(tmp_1st_loc < 0, (tmp_1st_loc), req_1st_loc + tmp_1st_loc)
    print(req_1st_loc.shape)
    # req_1st_loc = req_1st_loc + tmp_1st_loc if tmp_1st_loc < 0 else req_1st_loc
    tmp_1st_loc[tmp_1st_loc < 0] = 0
    tmp_1st_loc[tmp_1st_loc > 20] = 20

    tmp_2nd_loc = next_2nd_loc - req_2nd_loc + ret_2nd_loc
    # req_2nd_loc = req_2nd_loc - tmp_2nd_loc[tmp_2nd_loc < 0]
    req_2nd_loc = np.where(tmp_2nd_loc < 0, (tmp_2nd_loc), req_2nd_loc + tmp_2nd_loc )
    tmp_2nd_loc[tmp_2nd_loc < 0] = 0
    tmp_2nd_loc[tmp_2nd_loc > 20] = 20
    
    r_1st_loc = 10*req_1st_loc
    r_2nd_loc = 10*req_2nd_loc
    r = r_1st_loc + r_2nd_loc

    return r, tmp_1st_loc, tmp_2nd_loc

def policy_evaluation(policy, value_func, P, gamma=0.9, eps=10e-3):
    prev_value_func = np.copy(value_func)
    i=0
    while(np.max(np.abs(value_func-prev_value_func))<eps or i==0):
        prev_value_func = value_func
        r, next_1st_loc, next_2nd_loc = reward(policy)
        print(next_1st_loc)
        # print(next_2nd_loc)
        value_func = r + gamma*prev_value_func[next_1st_loc-1][next_2nd_loc-1]
    return value_func

def policy_improvement(policy, value_func):

    return policy

def policy_iteration():

    eps = 10e-3
    gamma = 0.9
    
    prev_policy = np.ones((20,20))
    policy = np.random.randint(-5,5,size=(20,20))
    value_func = np.zeros((20,20))
    
    while(np.max(np.abs(policy-prev_policy))>eps):
        value_function = policy_evaluation(policy, value_func, P=1)
        prev_policy = policy
        policy = policy_improvement(policy, value_func)
    
    return 0

if __name__ == "__main__":
    policy_iteration()