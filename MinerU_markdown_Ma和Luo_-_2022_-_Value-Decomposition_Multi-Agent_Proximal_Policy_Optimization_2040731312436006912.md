# Value-Decomposition Multi-Agent Proximal Policy Optimization

Yanhao Ma, Jie Luo 

College of Automation & College of Artificial Intelligence, Nanjing University of Posts and Telecommunications 

Nanjing 210023, China 

E-mail: ljnc@njupt.edu.cn 

Abstract—We explore the policy-based approach for the newly well-liked centralized training and decentralized execution (CTDE) mechanism’s multi-agent reinforcement learning (MARL) job. Multi-agent proximal policy optimization (MAPPO) achieved optimal effect in multiple multi-agent cooperative tasks. However, it performs poorly in more complex multi-agent cooperative tasks because it does not solve the problem of credit assignment. To deal with this issue, we combine the method of value decomposition in value-based reinforcement learning with policybased MAPPO, and propose a new actor-critic algorithm, namely value-decomposition multi-agent proximal policy optimization (VDPPO). VDPPO uses value decomposition to provide different rewards for agents to distinguish their contributions and train the policy network using the advantage function. To determine the effectiveness of our algorithm, we modify the multi-agent particle environments (MPE) and carry out experiments on it. The experimental findings demonstrate that our suggested method is more advantageous compared to several baseline algorithms and is suitable for more complex multi-agent cooperative tasks. 

Keywords—value decomposition, credit assignment, CTDE, VDPPO 

# I. INTRODUCTION

In the recent past, deep reinforcement learning (DRL) [1] has made groundbreaking progress, and numerous related algorithms and applications have emerged. DRL has achieved many successful applications in the single agent scenario, in which there is no need to model and predict other agents in the environment [2]. However, more application scenarios involve multiple agents, which must cooperate to achieve a common goal. For example, multi-robot control, distributed logistics and the coordination of autonomous vehicles. Obviously, the research of DRL in multi-agent control system is more challenging and valuable. 

The most naive method of MARL regards the other agents as a component of the environment and individually trains each agent to optimize its own return. For example, independent Qlearning [3] train an independent Q-learning model for each agent. However, Each agent’s policy will alter during training, creating a non-stationary environment. A simple and crude method to tackle this issue is to combine the information of all agents, the multi-agent system should be reduced to a single agent system, then use the single agent reinforcement learning algorithm for training. Although the centralized method solves the problem of stationarity, it also brings another serious problem, that is, curse of dimensionality. Therefore, CTDE mechanism, which can use global information for training and 

only use local observations for execution, has become a better choice. 

Lowe introduces deep deterministic policy gradient algorithm [4] from single agent domain into multi-agent domain and proposes a novel multi-agent algorithm, whose critic network can be trained by using global information [2]. On this basis, Iqbal proposes multi-agent soft Q learning in continuous action space to solve the problem of relative over generalization [5]. Chao Yu et al. applied proximal policy optimization (PPO) algorithm with multi-agent field, as well as proposed multi-agent proximal policy optimization algorithm [6], which achieved the optimal effect in multiple multi-agent cooperative tasks. These algorithms use additional information for training, which reduces the non-stationary of the environment, but does not solve the issue of credit assignment. 

Counterfactual multi-agent policy gradients [7] indicates the contribution of agents for team tasks by using counterfactual reasoning. Yali Du suggested that each agent learns an intrinsic reward function [8] so that agents are motivated differently even if the environment provides only one team reward. In recent years, the most popular research on credit assignment is value decomposition. The joint action-state value function is represented by the total of local action-state value functions in value decomposition networks [9], which allows learning to be centralized. However, it does not use additional status information. QMIX [10] uses non-negative mixing networks to represent a wider range of value decomposition functions. In addition, additional state information is captured by the hypernetwork that outputs the mixing network parameters. QTRAN [11] is a generalized factorization method, which can be applied to the environment without structural constraints. Weighted QMIX [12] improves QMIX focus on better joint actions by introducing weights. The Individual Global Max (IGM) principle is transformed into an advantage function constraint that is easy to implement using duplex dueling multiagent Q-learning (QPLEX) [13], so as to achieve efficient value function learning. 

Recently, scholars began to apply the value decomposition method based on value learning to the actor critic architecture. Value-decomposition multi-agent actor-critics [14] proposes a novel algorithm structure, which applies value function decomposition approach to the actor critic architecture. Factored multi-agent centralized policy gradients [15] applies value decomposition to MADDPG algorithm. Tianhao Zhang proposes 

to factorize the optimal policy generated by maximum entropy into individual policies [16] with reference to the idea of value decomposition. 

We suggest an innovative multi-agent algorithm called value decomposition proximal policy optimization, which combine the QPLEX algorithm based on advantage IGM with the MAPPO algorithm based on policy. VDPPO can alleviate the credit assignment problem of MAPPO algorithm, and it improves the training efficiency compared with the valuebased method because of the advanced actor-critic (A2C) architecture. VDPPO takes actor-critic architecture and adds local critics, which estimate the local action-state values. The central critic learns global action-state value is decomposed into local action-state values of each agent based on advantage IGM. The advantage functions offered by the local critics are used to train the policy network. 

# II. SYSTEM MODEL BUILDING

# A. Decentralized Partially Observable Markov Decision Processes (DEC-POMDP)

We use a DEC-POMDP defined by $\langle S , U , P , r , Z , O , N , \gamma \rangle$ to describe a fully cooperative task where $s \in S$ is the true state of the environment and $N$ is the set of agents that takes values from 1 to $n$ . At each time step, each agent $i \in N$ chooses an action $u _ { i } \in U$ forming a joint action $\pmb { u } \in \pmb { U } \equiv U ^ { n }$ . This causes a reward $r ( s , u ) : S \times U \to \mathbb { R }$ for the team and a transition to the next global state $s ^ { \prime }$ in accordance with the transition function $P ( s ^ { \prime } | s , u ) : S \times U \times S \  \ [ 0 , 1 ]$ . Each agent $i$ obtains its own partial observation $o _ { i } ~ \in ~ Z$ through the observation probability function $O ( s , i )$ owing to partial observability. Each agent learns a policy function $\pi _ { i } ( a _ { i } | \tau _ { i } )$ based solely on its own local action-observation history $\tau _ { i } \in T \equiv ( Z \times U ) ^ { * }$ and $\pmb { \tau } \in \pmb { T } \equiv T ^ { n }$ represent joint action-observation history. $\gamma ~ \in ~ [ 0 , 1 )$ is a discount factor. The formal objective is to identify a joint policy $\pi = \langle \pi _ { 1 } , \ldots , \pi _ { n } \rangle$ that maximizes the joint action-state value function $\begin{array} { r } { Q _ { t o t } ^ { \pi } ( s , \pmb { u } ) = \mathbb { E } [ \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r _ { l } | s , \pmb { u } ] } \end{array}$ or the joint value function $\begin{array} { r } { V _ { t o t } ^ { \pi } ( s ) = \mathbb { E } [ \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r _ { l } | s ] } \end{array}$ . 

# B. Value Decomposition

We utilize $Q ( \tau , { \boldsymbol { \mathbf { \mathit { u } } } } )$ rather than $Q ( s , { \pmb u } )$ owing to partial observability. Value decomposition shows the joint actionstate value function $Q _ { t o t } ( \pmb { \tau } , \pmb { u } )$ as a function of individual action-state value functions $[ Q _ { i } ( \tau _ { i } , u _ { i } ) ] _ { i = 1 } ^ { n }$ in order to identify the contributions of agents in cooperative tasks. IGM, which ensures that joint optimal actions are the same as individual optimal actions, is a key principle, i.e., 

$$
\underset {\boldsymbol {u}} {\operatorname {a r g m a x}} Q _ {t o t} = \left(\underset {u _ {1}} {\operatorname {a r g m a x}} Q _ {1}, \dots , \underset {u _ {n}} {\operatorname {a r g m a x}} Q _ {n}\right). \tag {1}
$$

The dueling architecture is suggested by the dueling deep Q network(DDQN) can be decomposed from the action-state value $Q$ into state value $V$ and advantage $A$ , i.e. $Q = V + A$ . Based on this idea, then $Q _ { t o t } ( \tau , u ) = V _ { t o t } ( \tau ) + A _ { t o t } ( \tau , u )$ and $Q _ { i } ( \tau , u _ { i } ) = V _ { i } ( \tau ) ~ + ~ A _ { i } ( \tau , u _ { i } )$ , where $\begin{array} { r l } { V _ { t o t } ( \tau ) } & { { } = } \end{array}$ $\operatorname* { m a x } _ { \boldsymbol { u ^ { \prime } } } Q _ { t o t } ( \boldsymbol { \tau } , \boldsymbol { u ^ { \prime } } )$ and $V _ { i } ( \pmb { \tau } ) = \operatorname* { m a x } _ { \ b { u _ { i } ^ { \prime } } } Q _ { i } ( \pmb { \tau } , \pmb { u _ { i } ^ { \prime } } )$ . Since state value 

$V$ is independent of action $u$ , the IGM constraint can be transformed into the constraint based on advantage $A$ : 

$$
\operatorname {a r g m a x} _ {\boldsymbol {u}} A _ {t o t} = \left(\operatorname {a r g m a x} _ {u _ {1}} A _ {1}, \dots , \operatorname {a r g m a x} _ {u _ {n}} A _ {n}\right). \tag {2}
$$

The constraint in (2) is equivalent to that when $\forall i \in N$ $\forall u ^ { * } \in U ^ { * }$ , $\forall u \in U \backslash U ^ { * }$ , 

$$
\left\{ \begin{array}{l} A _ {t o t} (\boldsymbol {\tau}, \boldsymbol {u} ^ {*}) = A _ {i} (\boldsymbol {\tau}, u _ {i} ^ {*}) = 0 \\ A _ {t o t} (\boldsymbol {\tau}, \boldsymbol {u}) \leqslant 0, A _ {i} (\boldsymbol {\tau}, u _ {i}) \leqslant 0 \end{array} , \right. \tag {3}
$$

where $\pmb { U } ^ { * } = \{ \pmb { u } | \pmb { u } \in \pmb { U } , Q _ { t o t } ( \pmb { \tau } , \pmb { u } ) = V _ { t o t } ( \pmb { \tau } ) \}$ . The use of advantage-based IGM under actor-critic structures not only enables consistency constraints directly by restricting the range of values for advantage functions, but also allows the value of advantage functions to be used to update policy networks. 

# C. VDPPO

MAPPO is a multi-agent setup extension of PPO. It is an actor-critic, on-policy approach that trains stochastic policies using the CTDE paradigm. Unlike MAPPO, which learns state value $V$ , to combine with the value decomposition approach, our method learns action-state value function $Q$ . A policy network $\pi _ { \theta }$ given the parameters $\theta$ and a critic network $Q _ { \phi }$ given the parameters $\phi$ are two independent neural networks that are trained to enable the learning of arbitrary reward functions. Because VDPPO can only access global information while undergoing centralized training, it’s critic is said to be centralized. The action-state value function $Q _ { \phi }$ is estimated using joint action $\textbf { \em u }$ and global state $s$ . In order to train the joint action-state value function, the following loss is minimized: 

$$
L (\phi) = \mathrm {E} _ {D} \left[ \max  \left(\left(y ^ {t o t} - Q _ {t o t} ^ {\phi} (\boldsymbol {\tau}, \boldsymbol {u})\right) ^ {2}, \left(y ^ {t o t} - Q _ {c l i p}\right) ^ {2}\right) \right], \tag {4}
$$

where $y ^ { t o t } = r ( s , \pmb { u } ) + \gamma Q _ { t o t } ^ { \phi } ( \pmb { \tau } ^ { \prime } , \pmb { u } ^ { \prime } )$ and $\begin{array} { r l } { Q _ { c l i p } } & { { } = } \end{array}$ $c l i p ( Q _ { t o t } ^ { \phi } ( \tau , { \pmb u } ) , Q _ { t o t } ^ { \phi _ { o l d } } ( \tau , { \pmb u } ) - \varepsilon , Q _ { t o t } ^ { \phi _ { o l d } } ( \tau , { \pmb u } ) + \varepsilon )$ . Here $D$ is the data buffer and $\phi _ { o l d }$ are old parameters before the update. $\varepsilon$ is a hyperparameter. 

The policy network is trained by maximizing the following objective 

$$
L (\theta) = \mathrm {E} _ {D} \left[ \min  \left(r _ {\theta}, r _ {\text {c l i p}}\right) A _ {i} \left(\boldsymbol {\tau}, u _ {i}\right) + \sigma S \left[ \pi_ {\theta} \left(\tau_ {i}\right) \right] \right], \tag {5}
$$

where rθ = $\begin{array} { r } { r _ { \theta } = \frac { \pi _ { \theta } \left( u _ { i } | \tau _ { i } \right) } { \pi _ { \theta _ { o l d } } \left( u _ { i } | \tau _ { i } \right) } } \end{array}$ and $r _ { c l i p } = c l i p ( r _ { \theta } , 1 - \varepsilon , 1 + \varepsilon )$ . Here $\sigma$ is the entropy coefficient hyperparameter and $S$ is the policy entropy. The advantage function $A _ { i } ( \tau , u _ { i } )$ is provided by individual critics. 

# III. METHOD

The goal of our work is to apply the value decomposition approach to the MAPPO algorithm. As a result, we suggest the novel multi-agent learning algorithm VDPPO, which is based on MAPPO. Fig. 1(b) depicts the overall architecture of VDPPO algorithm, which is made up of three primary parts: the network of individual action-state value, the dueling network and the network of individual policy. The network of individual action-state value provides differentiated rewards 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/12dfb367a822b91125bca0810d6220eb8dfe224f9f228ff3e658df9cae130e4e.jpg)



Fig. 1. (a) The structure of the dueling mixing network. (b) The VDPPO architecture overall. (c) The network structure of agent local critics. (d) Agent actor network structure.


for agents to distinguish their contributions in the team task. The dueling network integrates individual action-state values into the joint action-state value under the constraints of advantage-based IGM in order to train with team rewards. The network of individual policy is used to select the actions to be performed for agents. The network of individual actionstate value and the dueling network is trained to minimize Temporal-Difference errors while the policy network is trained by maximizing the advantage functions offered from the network of individual action-state value. During decentralized execution, individual action-state value network and dueling network will be removed and each agent will use its individual policy network to select actions according to local historical action observation data. 

# A. Individual Action-State Value Network

We provide each agent with a recurrent neural network for predicting local values that can accept the global state $s$ as input because it is only utilized during training. The recurrent Q-network shown in Fig. 1(c), which accepts previous hidden state $h _ { i } ^ { t - 1 }$ , current global state $s _ { t }$ , current local observations $o _ { i } ^ { t }$ , and previous action $u _ { i } ^ { t - 1 }$ as inputs and generates local $Q _ { i } ( \tau , u _ { i } )$ , represents the individual actionstate value of agents. According to the dueling structure of DDQN, the local $V _ { i } ( \tau ) ~ = ~ \operatorname* { m a x } _ { u _ { i } } Q _ { i } ( \tau , u _ { i } )$ and local $A _ { i } ( \tau , u _ { i } ) = Q _ { i } ( \tau , u _ { i } ) - V _ { i } ( \tau )$ can be derived separately. 

# B. Dueling Mixing Network

As showed in Fig. 1(a), the dueling mixing network takes the outputs $[ V _ { i } , A _ { i } ] _ { i = 1 } ^ { n }$ of the individual Q-network and global state $s$ as inputs, and generates $Q _ { t o t } ( \pmb { \tau } , \pmb { u } )$ . The dueling mixing network calculates the joint advantage $A _ { t o t } ( \pmb { \tau } , \pmb { u } )$ as well as the joint value $V _ { t o t } ( \tau )$ using individual dueling structures, and then uses the joint dueling architecture to produces the joint action-state value $Q _ { t o t } ( \pmb { \tau } , \pmb { u } )$ . 

Since the advantage-based IGM principle does not impose any constraints on the state value functions. Therefore, we 

calculate the joint value $V _ { t o t } ( \tau )$ using the same summation structure as QPLEX to facilitate easy and effective learning: 

$$
V _ {t o t} (\boldsymbol {\tau}) = \sum_ {i = 1} ^ {n} V _ {i} (\boldsymbol {\tau}) \tag {6}
$$

To ensure IGM consistency between individual advantage $[ A _ { i } ( \tau , u _ { i } ) ] _ { i = 1 } ^ { n }$ and the joint advantage $A _ { t o t } ( \pmb { \tau } , \pmb { u } )$ , the joint advantage function is calculated by VDPPO in the manner described below: 

$$
A _ {t o t} (\boldsymbol {\tau}, \boldsymbol {u}) = \sum_ {i = 1} ^ {n} \lambda_ {i} (\boldsymbol {\tau}, \boldsymbol {u}) A _ {i} (\boldsymbol {\tau}, u _ {i}), \tag {7}
$$

where $\lambda _ { i } ( \pmb { \tau } , \pmb { u } ) > 0$ . The positivity brought on by $\lambda _ { i } ( \pmb { \tau } , \pmb { u } )$ will ensure that the maximum joint action-state value corresponds to the same action as the maximum individual action-state value. The joint information of $\lambda _ { i } ( \pmb { \tau } , \pmb { u } )$ provides sufficient expressiveness for value decomposition. With (6) and (7), $Q _ { t o t } ( \pmb { \tau } , \pmb { u } )$ can be rewritten as 

$$
Q _ {t o t} (\boldsymbol {\tau}, \boldsymbol {u}) = \sum_ {i = 1} ^ {n} \left[ Q _ {i} (\boldsymbol {\tau}, u _ {i}) + \left(\lambda_ {i} (\boldsymbol {\tau}, \boldsymbol {u}) - 1\right) A _ {i} (\boldsymbol {\tau}, u _ {i}) \right] \tag {8}
$$

To minimize the loss as (4), individual action-state value network and dueling mixing network are trained end-to-end. 

# C. Individual Policy Network

We designed a recurrent neural network with parameters $\theta$ that selects actions for agents, as shown in Fig. 1(d). The network takes local observations and historical actions as inputs and is trained to optimize the target as (5). We demonstrate the entire training process in algorithm 1 to further illustrate the VDPPO algorithm. 

# IV. EXPERIMENTS

In this section, we present our experimental environment in accordance with multi-agent particle environment proposed in [2]. Then, we compare our algorithm with the most advanced 


Algorithm 1 VDPPO


1: Initialize critic $\phi$ , target critic $\phi'$ and actor $\theta$ 2: for step = 1 to max_step do  
3: set empty buffer  
4: for $e_c = 1$ to batch_size do  
5: initialize actor RNN states $h_1^0, \ldots, h_n^0$ 6: for $t = 1$ to $T$ do  
7: for all agent $i$ do  
8: $p_i^t, h_i^t = \pi(o_i^t, u_i^{t-1}, h_i^{t-1}, \theta)$ 9: Sample action $u_i^t$ from $p_i^t$ 10: end for  
11: Receive reward $r_t$ and transfer to the state $s_{t+1}$ 12: end for  
13: save experience in buffer  
14: end for  
15: for $t = 1$ to $T$ do  
16: Calculate $y^{tot}$ using $\phi'$ 17: Calculate advantage estimate $A_i$ using $\phi$ 18: end for  
19: Update $\phi$ on L( $\phi$ ) using (4)  
20: Update $\theta$ on L( $\theta$ ) using (5)  
21: Every $C$ steps update target critic $\phi'$ using $\phi$ 22: end for 

value-based methods QMIX [10], QPLEX [13], multi-agent actor-critic algorithm MAPPO [6], and achieve the optimal performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/bd08a831732ca7f7614980becc00b6800c09a31ac1268d3c219a65342ea5b4b6.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/3c36a5a984abe179378d6dfe9358b892f5f0db06e4a21e2ddae563576e8d0116.jpg)



(b)



Fig. 2. Predator-Prey experimental scenario


# A. Environments

We use MPE, a two-dimensional particle environment with discrete time and continuous space that has $N$ agents and $L$ obstacles, to carry out our experiments. As shown in Fig. 2 and Fig. 3, black circles represent obstacles and other circles represent agents. Agents can perform physical actions to interact with the environment. We substitute a pre-trained neural network to make decisions for the prey to create a fully cooperative environment. We have experimented with a cooperative variant of the simple tag scenario in MPE. We give specifics for each environmental scenario below. 

a) Predator-Prey: There are $N$ slower cooperating agents, a speedier adversary and $L$ huge obstacles used to 

obstruct the way in this game. The goal of agents is to catch the adversary as fast as possible while the adversary evades cooperating agents. Just like depicted in 2 Fig. 2(a), Three red agents round up a purple adversary in a field of three black obstacles. Each time a cooperative agent collides with an adversary, as shown in Fig. 2(b), all agents receive a team reward. Agents can observe the environment to obtain partial information, including the absolute speed of other agents and the adversary, as well as the relative positions of other objects. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/07774af2f633d75cad4ef9ffb6b8f92ea6f249d592dff8d0c5910ce2da5fe139.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/8263241ef5eec12140a972ba8add7f2257d74941e5287051c1389b3cd09b142d.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/201e752b676f339e6bd053074c518a241543b9c320f9de45e436937ef5781b71.jpg)



(c)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/37d9ee2e883e7fd831c7070397264d7e46c015482d613e5f93df8ed3f770a772.jpg)



(d)



Fig. 3. Modified Predator-Prey experimental scenario


b) Modified Predator-Prey(MPP): By modifying the reward function in the predator-prey game, we obtain a more complex environment whose task is set to be non-monotonic. The construction of state space and action space is similar to the predator-prey scenario. As shown in Fig. 3(a), to increase the complexity of the task, We add a captured range for the adversary, where little blue circle represents the adversary and light blue circle represents its captured range. When an agent enters the captured range of the adversary we call the adversary captured. We extend it to the scenario where positive rewards are given only when multiple agents capture an adversary at the same time, requiring a higher degree of cooperation. Agents will get a positive team reward if two or more agents capture an adversary at the same time, Just like depicted in Fig. 3(c) and Fig. 3(d). Agents will receive a negative team reward when only one agent catches the adversary as shown in Fig. 3(b). 

# B. Results

We integrate all local observations to create the global state since MPE does not provide the global environmental state 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/6de7d3f030dcab2da0229ef68741999814df14304f2ae20a4e4114b281428bb8.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/bcdc75a5d8202789cb3043619d666aa075cc841d2babbe6f90d3d78226e5bc97.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-05/2be7a77b-f960-4038-8ecf-ce3b978da2dd/5f70e2a564269e6ff3bea203843beec6b7fbdce7e290f775c8f7fab098faa81e.jpg)



(c)



Fig. 4. (a) Average reward per episode in Predator-prey scenario. (b) Average reward per episode in the MPP scenario. (c) Average reward per episode of MAPPO algorithm in MPP scenario. The mean through five seeds is drawn and the value range is shown shaded. As the MAPPO algorithm curve in (b) is not obvious, it is drawn separately in (c).


and has a low observation dimension. All results are averaged over 5 seeds. 

a) Predator-Prey: Fig. 4(a) shows the average episode returns obtained by the different methods in the predatorprey task. It can be seen that VDPPO outperforms the other algorithms in terms of absolute performance and learning efficiency. 

b) MPP: We plot the average episode returns curves of the four algorithms in the MPP experimental scenario in Fig. 4(b). The results show that VDPPO is superior to other algorithms. Due to the presence of penalty items, MAPPO agents learn to work together with limited coordination, so they do not actively try to capture the adversary, but rather try to minimize the risk of penalty. As shown in Fig. 4(c), MAPPO do not cooperate at all and only learn a sub-optimal policy: escape from the adversary to minimize the risk of punishment. The results of this experiment show that value decomposition can improve MAPPO’s ability to deal with complex cooperative tasks. 

# V. CONCLUSION

In this paper, we use value decomposition to improve the MAPPO algorithm and present a novel algorithm VDPPO. VDPPO takes advantage of CTDE to solve the credit assignment problem by appropriately converting and decomposing joint action-state value into individual action-state values. From the experimental results, the algorithm improves MAPPO’s performance in cooperative tasks, achieving optimal performance in the multi-agent predator-prey environment compared to several baseline algorithms. Our approach can be used to solve more complex multi-agent cooperative tasks. The experimental results demonstrate the convergence of our algorithm, and in the future work we hope to prove the convergence of the algorithm theoretically. 

# REFERENCES



[1] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath, “Deep reinforcement learning: A brief survey,” IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 26–38, 2017. 





[2] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multiagent actor-critic for mixed cooperative-competitive environments,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 6382–6393. 





[3] M. Tan, “Multi-agent reinforcement learning: Independent vs. cooperative agents,” in Proceedings of the tenth international conference on machine learning, 1993, pp. 330–337. 





[4] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning.” in International Conference on Learning Representations, 2016. 





[5] S. Iqbal and F. Sha, “Actor-attention-critic for multi-agent reinforcement learning,” in International conference on machine learning. PMLR, 2019, pp. 2961–2970. 





[6] C. Yu, A. Velu, E. Vinitsky, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of ppo in cooperative, multi-agent games,” arXiv preprint arXiv:2103.01955, 2021. 





[7] J. N. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, “Counterfactual multi-agent policy gradients,” in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, 2018, pp. 2974–2982. 





[8] Y. Du, L. Han, M. Fang, T. Dai, J. Liu, and D. Tao, “Liir: learning individual intrinsic reward in multi-agent reinforcement learning,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems, 2019, pp. 4403–4414. 





[9] P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg et al., “Value-decomposition networks for cooperative multi-agent learning based on team reward,” in Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, 2018, pp. 2085–2087. 





[10] T. Rashid, M. Samvelyan, C. Schroeder, G. Farquhar, J. Foerster, and S. Whiteson, “Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning,” in International conference on machine learning. PMLR, 2018, pp. 4295–4304. 





[11] K. Son, D. Kim, W. J. Kang, D. E. Hostallero, and Y. Yi, “Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning,” in Proceedings of the 31st International Conference on Machine Learning, Proceedings of Machine Learning Research. PMLR, 2019. 





[12] T. Rashid, G. Farquhar, B. Peng, and S. Whiteson, “Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning,” Advances in neural information processing systems, vol. 33, pp. 10 199–10 210, 2020. 





[13] J. Wang, Z. Ren, T. Liu, Y. Yu, and C. Zhang, “Qplex: Duplex dueling multi-agent q-learning,” in International Conference on Learning Representations, 2020. 





[14] J. Su, S. Adams, and P. Beling, “Value-decomposition multi-agent actorcritics,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 13, 2021, pp. 11 352–11 360. 





[15] B. Peng, T. Rashid, C. Schroeder de Witt, P.-A. Kamienny, P. Torr, W. Bohmer, and S. Whiteson, “Facmac: Factored multi-agent centralised ¨ policy gradients,” Advances in Neural Information Processing Systems, vol. 34, pp. 12 208–12 221, 2021. 





[16] T. Zhang, Y. Li, C. Wang, G. Xie, and Z. Lu, “Fop: Factorizing optimal joint policy of maximum-entropy multi-agent reinforcement learning,” in International Conference on Machine Learning. PMLR, 2021, pp. 12 491–12 500. 

