# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.

## POLICY IMPROVEMENT FUNCTION
### Name: DHEENA DARSHINI KARTHIK DHEEPAN
### Register Number: 212223240030
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

pi_2 = policy_improvement(V1, P)
print("Name: Prasannalakshmi G")
print("Register Number: 212222240075")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

```
## POLICY ITERATION FUNCTION
### Name: DHEENA DARSHINI KARTHIK DHEEPAN
### Register Number: 212223240030
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]

  while True:
    old_pi = {s: pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P, gamma, theta)
    pi = policy_improvement(V, P, gamma)

    if old_pi == {s: pi(s) for s in range(len(P))}:
      break

  return V, pi
optimal_V, optimal_pi = policy_iteration(P)
print("Name: Prasannalakshmi G")
print("Register Number: 212222240075")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
#### POLICY:
<img width="1011" height="289" alt="Screenshot 2025-09-27 082550" src="https://github.com/user-attachments/assets/e5bebb84-c340-41a0-90c4-fad7416b39ad" />


#### STATE VALUE FUNCTION:
<img width="1022" height="279" alt="Screenshot 2025-09-27 082623" src="https://github.com/user-attachments/assets/33fa64d0-c642-400f-83fc-c750afa7c620" />


#### SUCCESS:
<img width="1220" height="73" alt="Screenshot 2025-09-27 082643" src="https://github.com/user-attachments/assets/ecaf1005-502a-4496-9703-04e5513c4161" />


### 2. Policy, Value function and success rate for the Improved Policy
#### POLICY:
<img width="1007" height="288" alt="Screenshot 2025-09-27 082854" src="https://github.com/user-attachments/assets/d03fca67-ca04-4d57-ac7c-484223933ecc" />


#### STATE VALUE FUNCTION:
<img width="1067" height="281" alt="Screenshot 2025-09-27 082906" src="https://github.com/user-attachments/assets/a34d94a9-88b4-4305-b599-b9ecc203729f" />


#### SUCCESS:
<img width="1220" height="66" alt="Screenshot 2025-09-27 083042" src="https://github.com/user-attachments/assets/2e28c976-eb61-4fe1-aa7d-adc36c8ade15" />

<img width="883" height="80" alt="Screenshot 2025-09-27 083054" src="https://github.com/user-attachments/assets/7c392880-50aa-446f-94ed-82c7a79bc80a" />



### 3. Policy, Value function and success rate after policy iteration
#### POLICY:
<img width="1020" height="325" alt="Screenshot 2025-09-27 083136" src="https://github.com/user-attachments/assets/349fff70-cdb4-417c-be1a-b1203ca34824" />


#### STATE VALUE FUNCTION:
<img width="982" height="275" alt="Screenshot 2025-09-27 083155" src="https://github.com/user-attachments/assets/ecf068f3-57e7-4702-8075-fd2dc2996225" />

#### SUCCESS:
<img width="1221" height="79" alt="Screenshot 2025-09-27 083203" src="https://github.com/user-attachments/assets/3d2f0983-2baa-423a-b269-6c087f3a685f" />




## RESULT:
Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
