# Basic concepts
### Markov Decision Process (MDP)
### State
### State Space
the set of states $\mathcal{S}$
### Action
### Action Space: 不同状态下action space不一样
the set of actions $\mathcal{A}(s)$ is associated for state $s \in \mathcal{S}$.
### state transition
$$
s_1 \xrightarrow{a_2} s_2
$$

### state transition probability: 
$$
p(s_2 \mid s_1, a_2) = 1
$$

$$
p(s_i \mid s_1, a_2) = 0 \quad \forall i \ne 2
$$

### Policy: tell the agent what actions to take at a state

environment 本身不改变，给定某个状态和 agent 的操作后，环境会给出相应的响应 $p(r_{t+1} \mid s_t,a_t) \quad p(s_{t+1} \mid s_t,a_t)$；Policy $\pi$ 指的是在某个状态下 agent 怎么操作

因此当 $(s_t, a_t)$ 给定, then $r_{t+1}$ and $s_{t+1}$ 与策略无关，仅由环境响应决定；

deterministic policy 
$$
\pi(a_1 \mid s_1) = 0,\  \pi(a_2 \mid s_1) = 1
$$
stochastic policy
$$
\pi(a_1 \mid s_1) = 0.5, \ \pi(a_2 \mid s_1) = 0.5
$$

### Reward
**a real number we get after taking an aciton**
**a human-machine interface with which we can guide the agent to behave as what we experct**
the set of rewards $\mathcal{R}(s, a)$
Mathematical  description:  $p(r = -1 \mid s_1, a_1) = 1, \  p(r \neq -1 \mid s_1, a_1) = 0$
即在s1采取a1获得reward -1的概率是1，此时就是确定性的
下标习惯上记作  $S_t \xrightarrow{A_t} R_{t+1}, S_{t+1}$ ，表示状态 St 采取 At 的Reward是 $R_{t+1}$

### Trajectory: a state-action-reward chain
$$
s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9
$$
trajectory 可以 infinite
比如到了某个状态每次采取的行动就是保持不动，那么就一直循环下去，并且这个状态下保持不动每次的 reward 为1，为了让这个 infinite trajectory 的 return 收敛，要引入 discount $\gamma \in [0,1)$
$$
\begin{aligned}
\text{discounted return} 
&= 0 + \gamma 0 + \gamma^{2} 0 + \gamma^{3} 1 + \gamma^{4} 1 + \gamma^{5} 1 + \dots \\
&= \gamma^{3} (1 + \gamma + \gamma^{2} + \dots) \\
&= \gamma^{3}\frac{1}{1-\gamma}
\end{aligned}
$$

一方面数学上收敛，另一方面 discount rate 本身也是一种对过去和未来 reward 的加权

### Return
return 是针对 trajectory 而言的
The return of a trajectory is the sum of **all the rewards collected along the trajectory**

### Episode tasks/ Continuing tasks
**Episode: 从环境重置开始到终止为止的一整段交互过程**。the agent stop at some terminal state
Continuing tasks: 永远持续下去的交互过程

Episode 可以转换成 continuing tasks 去看待：
- target state 搞成 absorbing state，一旦 agent 进入，就呆在里面不出来，后续每次 r = 0
- target state 视作一个 normal state， agent 进入后可以离开

### **Markov property**: memoryless property
$$
p(s_{t+1} \mid a_{t+1}, s_t, \dots, a_1, s_0)= p(s_{t+1} \mid a_{t+1}, s_t)
$$

$$
p(r_{t+1} \mid a_{t+1}, s_t, \dots, a_1, s_0)= p(r_{t+1} \mid a_{t+1}, s_t)
$$

**强化学习过程中 agent 和环境的交互过程是一个 Markov 过程**


---

### RL 训练框架 (Episode index / Episode length)

环境是没法改变的，训练是通过尝试，不断调整与环境交互的策略；比如 LOL，环境指的就是游戏本身(游戏代码)，agent 是玩家，玩家改变策略，游戏代码本身是不变的;

**episode 起始点**：环境 reset 给一个初始状态分布 $p(s_0)$ 。比如走迷宫放回左上角。
**episode 终止点**：到达目标、掉坑、失败、超时等

从起始点到终点，得到一个 episode，这个过程中做了多少 action，就是这个 episode 的 length ， 相应 episode index += 1；然后 reset 开始下一个 episode


每个 episode 内部，通过某种方式获取
$\left( (s_0,a_0,r_1,s_1),\ (s_1,a_1,r_2,s_2),\ \ldots,\ (s_{T-1},a_{T-1},r_T,s_T) \right)$ 用于更新参数和 policy

**单步 experience**：一次 transition $(s_t, a_t, r_{t+1}, s_{t+1}, done_t)$ 

后面 MC learning 提到的 first-visit 和 every-visit 只是在一个 episode 内获取 experience 的不同方式




----



# Bellman equation
## Notations
单步过程表示为: $S_t \xrightarrow{A_t} R_{t+1}, S_{t+1}$
- $R_{t+1}$: the reward obtained after taking $A_t$  
- $S_{t+1}$: the state transited to after taking $A_t$

$S_t, A_t, R_{t+1}$ 都是**随机变量**

This step is governed by the following probability distributions:
- $S_t \rightarrow A_t$ is governed by $\pi(A_t = a \mid S_t = s)$
- $S_t, A_t \rightarrow R_{t+1}$ is governed by $p\,(R_{t+1} = r \mid S_t = s, A_t = a)$
- $S_t, A_t \rightarrow S_{t+1}$ is governed by $p\,(S_{t+1} = s' \mid S_t = s, A_t = a)$

## State Value
A multi-step trajectory: 
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \xrightarrow{A_{t+2}} R_{t+3}, \ldots
$$
The **discounted return** is: 
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots
$$

- $\gamma \in [0, 1)$
- $G_t$ is also a random variable since $R_{t+1}, R_{t+2}, \ldots$ are random variables.

**state value**: 
$$
v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]
$$
如果某个状态的 state value 大，说明这个状态值得我们朝那个方向走
- 与当前状态 $s$ 有关，也与policy $\pi$ 有关，评估**当前策略 $\pi$ 下，某个 state 的价值**

**Return 和 state value 的关系**
return 是单个 trajectory 意义下的； state value 是概率意义下很多可能的 trajectory 的期望；
如果从某个状态出发的 trajectory 是确定性的，那么此时的 return 等价于 state value

## 条件期望进阶回顾
概率的**chain rule**:
$$
P(A_1 A_2 \cdots A_n)
= P(A_1)P(A_2 \mid A_1)P(A_3 \mid A_1A_2)\cdots
P(A_n \mid A_1\cdots A_{n-1})
$$
之前右侧合并都是从左往右合并，实际上不一定非要这样，把公共的条件当成没有，先忽略，最后再补上，**没条件也是一种特殊的条件**: 
$$
P(A_2 \mid A_1)P(A_3 \mid A_1A_2) = P(A_2A_3 \mid A_1)
$$

全概率公式两边取期望，就得到全期望公式；期望只是一个算子，只要某个东西是随机变量，自然可以对他取期望:
$$
P(X = x) = \sum_{y} P(X = x \mid Y = y)\,P(Y = y)
$$
$$
E[X] = \sum_y E[X \mid Y = y]\,P(Y = y).
$$
公共条件先忽略，最后再补上:
$$
P(X = x \mid Z = z) = \sum_{y} P(X = x \mid Y = y, Z = z)\,P(Y = y \mid Z = z)
$$
全期望公式直接从期望定义角度推本质上也就是一个二重积分换序
$$
\mathbb{E}[X \mid Z = z]
  = \sum_{y} \mathbb{E}[X \mid Y = y, Z = z] \, P(Y = y \mid Z = z)
$$

## Bellman equation 基本推导思想

**把某个具体的 state value 用下一步可能的 state value 表示**
$\forall \, s \in S$, 有,
$$
\begin{aligned}
v_{\pi}(s)
&= \mathbb{E}[G_t \mid S_t = s] \\ 
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\ 
&=\mathbb{E}[R_{t+1} \mid S_t = s]
  + \gamma \,\mathbb{E}[G_{t+1} \mid S_t = s] \\
&=  r_\pi(s) + \sum_{s'} \mathbb{E}[G_{t+1}\mid S_t = s,\, S_{t+1} = s']\, \, p_\pi(s' \mid s) \\
&= r_\pi(s) + \sum_{s'} \mathbb{E}[G_{t+1}\mid S_{t+1} = s']\, \, p_\pi(s' \mid s) \\
&= r_\pi(s) + \gamma \sum_{s'} p_\pi(s' \mid s)\, v_\pi(s')
\end{aligned}
$$
推导过程无非就是 $\mathbb{E}[R_{t+1} \mid S_t = s]$ 和 $\mathbb{E}[G_{t+1} \mid S_t = s]$  前者记作 $r_\pi(s)$ ,为 s 状态下进行操作 a 得到的 reward，这一项把此时的 a 空间当成边缘去积分；$G_{t+1}$ 涉及下一个状态的信息，这一项把下一步可能的状态空间 $S'$ 当成边缘去积分。对 $S$ 中的每个状态都计算上式，解方程组

## Bellman equation 的矩阵形式
$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$
以 $|\mathcal{S}|=4$ 为例：
$$
\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}
=
\begin{bmatrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
r_\pi(s_3) \\
r_\pi(s_4)
\end{bmatrix}
+\gamma
\begin{bmatrix}
p_\pi(s_1 \mid s_1) & p_\pi(s_2 \mid s_1) & p_\pi(s_3 \mid s_1) & p_\pi(s_4 \mid s_1) \\
p_\pi(s_1 \mid s_2) & p_\pi(s_2 \mid s_2) & p_\pi(s_3 \mid s_2) & p_\pi(s_4 \mid s_2) \\
p_\pi(s_1 \mid s_3) & p_\pi(s_2 \mid s_3) & p_\pi(s_3 \mid s_3) & p_\pi(s_4 \mid s_3) \\
p_\pi(s_1 \mid s_4) & p_\pi(s_2 \mid s_4) & p_\pi(s_3 \mid s_4) & p_\pi(s_4 \mid s_4)
\end{bmatrix}
\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}.
$$

可见 $P_\pi(i,j)$ 为 $s_i$ 向 $s_j$ 的转移概率

**解 $v_\pi$ 的目的是Policy Evaluation**
 
直接解线性方程组或者**迭代**，随机给一个初始值 $v_0 \in R^{|S|}$, 按照 $v_{k+1} = r_\pi + \gamma P_\pi v_k$ 迭代, 可以证明：$$\lim_{k \to \infty} v_k = v_\pi$$

## Action value
回顾：
A multi-step trajectory: 
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \xrightarrow{A_{t+2}} R_{t+3}, \ldots
$$
The **discounted return** is: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
那么有
State value: 从一个状态出发，return 的期望
$$
\begin{aligned}
v_{\pi}(s)
&= \mathbb{E}[G_t \mid S_t = s] \\ 
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
\end{aligned}
$$
Action value: 从一个状态出发，采取某种行动，return 的期望; 评估**在当前策略 $\pi$ 下，在某个 state 选某个 action 的价值**
$$q_\pi(s,a) = \mathbb{E}\big[ G_t \mid S_t = s, A_t = a \big]$$显然，$$v_{\pi}(s) = \sum_{a} \pi(a \mid s)\, q_{\pi}(s,a)$$知道所有的 state value 可以解得所有 action value, and vice versa

注意，某个状态采取某个措施是基于policy的，而 state value 和 action value 是评估当前 policy 的，假设当前 policy 下在 $s_1$ 只能 $a_1$ , $q_\pi(s_1,a2)$ 也是有意义的

# Bellman Optimality Equation (BOE)

## Optimal Policy

Optimal Policy $\pi^*$ : 
$$
v_{\pi^*}(s) \ge v_{\pi}(s), \quad\forall \pi, \ \forall s \in S
$$

有了这个最优策略的定义，带来以下问题
- 最优是否存在，是否唯一
- 最优策略是 stochastic 还是 deterministic
- 如何获得最优策略
用 BOE 可以解决以上问题

最大化全状态空间 state value 时对应的 policy 即为最终目标，优化问题描述为:
$$
\begin{aligned}
v &= \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v \right) \\
& = \max_{\pi} \sum_{a} \pi(a \mid s)\, q(s,a)
\end{aligned}
$$
其中 $v, r_{\pi}$ 等都是向量,  $\sum \pi=1, \pi>0$，我们可以知道最值一定如下(直观上就是给最大的 action value 的权重搞成1)
$$
\max_{\pi} \sum_{a} \pi(a \mid s)\, q(s,a)
= \max_{a \in \mathcal{A}(s)} q(s,a)
$$
问题是我们现在知道最值就是这个 $\max_{a \in \mathcal{A}(s)} q(s,a)$ ，但是具体什么样子的 $\pi$ 下取到这个最值，也就是还是要求具体的最优策略是什么

$v = \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v \right)$ , 令 $f(v)=\max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v \right)= \max_{\pi} \sum_{a} \pi(a \mid s)\, q(s,a)$ ，这就是一个**压缩映射的不动点问题**,  从随机策略出发，按照如下方式迭代，最终$v$ 收敛, 对应 $π$ 自然也收敛；
$$
 \pi_{k+1}(s) \in \arg\max_{a \in \mathcal{A}(s)} q^{\pi_k}(s,a),\ \forall s \in \mathcal{S}
$$

其中 **$\pi_{k+1}(s)$ 是一个确定性策略，用 $\in$ 是因为可能存在多个 action，都达到 action value 最大值，任取其一即可**
最终得到：
$$
v^* = r_{\pi^*} + \gamma P_{\pi^*} v^*
$$
并且可以证明**这个收敛得到的策略就是最优策略**:
对应的 deterministic policy 如下

$$
\pi^*(a \mid s) =
\begin{cases}
1, & a = a^*(s) \\
0, & a \ne a^*(s)
\end{cases}
$$

where
$$
a^*(s) = \arg\max_a q^*(a,s),
$$

**Theorem (Optimal policy invariance).** Consider a Markov decision process with $v^* \in \mathbb{R}^{|S|}$ as the optimal state value satisfying $v^* = \max_{\pi \in \Pi} (r_\pi + \gamma P_\pi v^*)$. If every reward $r \in \mathbb{R}$ is changed by an affine transformation to $\alpha r + \beta$, where $\alpha, \beta \in \mathbb{R}$ and $\alpha > 0$, then the corresponding optimal state value $v'$ is also an affine transformation of $v^*$:
$$
v' = \alpha v^* + \frac{\beta}{1-\gamma} 1,
$$
where $\gamma \in (0,1)$ is the discount rate and $1 = [1,\ldots,1]^T$. 因此, $r$ 仿射，对应 $v$ 也有一个仿射，虽然第二个系数不一样，但是选最优策略只看相对排序，$v$ 的仿射不改变相对大小顺序，有 policy invariance.



---


## 泛函简单知识回顾

**Linear Operator**：把某个向量空间投影到另一个向量空间的算子，且满足
$$
A(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha A \mathbf{x} + \beta A \mathbf{y}
$$

**Operator norm**：衡量两个赋范线性空间之间的有界线性算子的大小。
有界线性算子A (E到F)连续，当且仅当存在常数c，$\forall u \in E$ , 有:
$$
|A(u)\|_{F} \le c \cdot \|u\|_{E}.
$$
该定义说明，连续线性算子将E中的向量映射到F中时，对应的范数数值不会超过c倍；把算子范数定义为上界 c 中最小的一个
$$
|A\| = \inf \{\, c\ ;\ \|A(u)\|_{F} \leqslant c \cdot \|u\|_{E},\ \forall u \in E \}
$$
以下都是等价定义:
$$
\begin{aligned}
\|A\|
&= \sup\left\{\|Av\|_{F} : \|v\|_{E} \le 1 \right\} \\
&= \sup\left\{\|Av\|_{F} : \|v\|_{E} < 1 \right\} \\
&= \sup\left\{\|Av\|_{F} : \|v\|_{E} \in \{0,1\} \right\} \\
&= \sup\left\{\|Av\|_{F} : \|v\|_{E} = 1 \right\} \\
&= \sup\left\{\frac{\|Av\|_{F}}{\|v\|_{E}} : v \neq 0 \right\}
\end{aligned}
$$
因此**算子范数是绑定向量空间 $\mathcal{E}$ 定义的范数和向量空间 $\mathcal{F}$ 定义的范数**

对于欧式空间(向量为2范数)到欧式空间的矩阵算子 $\mathbf{A} : \mathbb{R}^n \to \mathbb{R}^m$
$$
\|A\| := \sup_{\|x\|_2=1}\|Ax\|_2
= \sup_{\|x\|_2=1}\sqrt{x^\top A^\top A x}.
$$

令 $B := A^\top A$，则 $B$ 为对称矩阵，可以转换为 Rayleigh quotient  
$$
\|A\|^2
= \sup_{\|x\|_2=1} x^\top B x
= \sup_{x\neq 0}\frac{x^\top B x}{x^\top x},
$$
这样得到的矩阵A的范数叫做 **induced 2-norm**，如下
$$
\|A\|_{2}
= \sqrt{\lambda_{\max}\!\left(A^\top A\right)}
= \sigma_{\max}(A)
$$


---


# Value iteration and policy iteration
## Value iteration algorithm
BOE 中给出了优化问题，
$$
\begin{aligned}
v &= \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v \right) \\
& = \max_{\pi} \sum_{a} \pi(a \mid s)\, q(s,a)
\end{aligned}
$$
这个式子作为定义式，不依赖于模型，始终都是最原始最重要的，其他的次级形式基于某些条件再具体推得即可
迭代 $\pi_k$ 是为了让 $v_{k+1}=f(v_{k})= \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v_k \right)$ , 从而由 contraction mapping 可 得 $v_k$ 收敛

**实现思路**为:
$$
\begin{align*}
v_k(s) &\rightarrow \arg\max_a q_k(s,a)\\
       &\rightarrow \text{greedy policy }\pi_{k+1}(a \mid s) \\
       &\rightarrow \text{new value } v_{k+1} = \max_a q_k(s,a)
\end{align*}
$$
### Algorithm 4.1: Value iteration algorithm
**Initialization:**  $p(r \mid s,a)$ and $p(s' \mid s,a)$ for all $(s,a)$ are known and **independent of the agent's policy $\pi(a \mid s)$** .  *Initialize $v_0$ arbitrarily.*
**Goal:**  When $\lVert v_k - v_{k-1} \rVert$ is greater than a predefined small threshold, for the $k$-th iteration, 
- For every state $s \in \mathcal{S}$, do  
  - For every action $a \in \mathcal{A}(s)$, do  
    - q-value:  $q_k(s,a) = \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a) v_k(s')$
    - Maximum action value:  $a_k^*(s) = \arg\max_a q_k(s,a)$
    - Policy update:  $\pi_{k+1}(a \mid s) = 1 \text{ if } a = a_k^*, \text{ and } \pi_{k+1}(a \mid s) = 0 \text{ otherwise}$
    - Value update:  $v_{k+1}(s) = \max_a q_k(s,a)$

**这段 pseudocode 中 $v_k$ 不是 state value，他只是一个迭代量，不满足 Bellman 公式，只是最终会收敛到 optimal state value 而已**

## Policy iteration algorithm
### Algorithm 4.2: Policy iteration algorithm
**Initialization:** $p(r \mid s,a)$ and $p(s' \mid s,a)$ for all $(s,a)$ are known. *Initialize $\pi_0$ arbitrarily.*  
**Goal:** When $v_{\pi_k}$ has not converged, for the $k$th iteration, do 
- *Policy evaluation:*  
  - Initialization: an arbitrary initial guess $v_{\pi_k}^{(0)}$.  
  - While $v_{\pi_k}^{(j)}$ has not converged, for the $j$th iteration, do  
    - For every state $s \in S$, do  
      -   $v_{\pi_k}^{(j+1)}(s) = \sum_a \pi_k(a \mid s)\left[ \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a)\, v_{\pi_k}^{(j)}(s') \right]$
- *Policy improvement:*  
  - For every state $s \in S$, do  
    - For every action $a \in A$, do  
      - $q_{\pi_k}(s,a) = \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a)\, v_{\pi_k}(s')$  
      - $a_k^*(s) = \arg\max_a q_{\pi_k}(s,a)$  
      - $\pi_{k+1}(a \mid s) = 1$ if $a = a_k^*$, and $\pi_{k+1}(a \mid s) = 0$ otherwise

**这段 pseudocode 无非是每次确定一个策略后，迭代法解 Bellman 公式，确切得到此时的 state value**

## Truncated policy iteration
这个是一般形式，value iteration 和 policy iteration 都是他的特殊形式，三者关系图见“西湖大学RL，第4课08:50”
实际执行也不可能无限迭代，其实 policy iteration algorithm 的具体实现就是 Truncated policy iteration



---

# Monte Carlo Learning

#### 什么是 model-free

**Environment Model.** MDP 的完整数学描述，包括 $p(s' \mid s,a)$ 和 $p(r \mid s,a)$；描述的是“世界怎么响应你的动作”
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \xrightarrow{A_{t+2}} R_{t+3}, \ldots
$$
**Model-Based.** agent 学习出 Environment Model，然后利用进行 Environment Model 中 MDP 的数学描述，完成 $q_{\pi_k}(s,a)$ 的计算，再进行规划
**Model-Free.** agent 通过采样 Experience 绕过 Environment Model

知道 model 其实就是知道环境对 agent 操作的响应机制, agent 做出某个操作，我们可以直接算得环境如何响应 (比如算出确定性相应或者算出响应的随机分布)，而不用去采样拟合

**Model 和 Policy 的关系**：model 本身不包含策略 $\pi$ ， model 描述的是“世界怎么相应你的动作”；策略 $\pi$ 描述 "在某个状态如何选动作"

#### MC learning
要么用模型，要么用数据；
原本是基于模型，然后计算 $q_{\pi_k}(s,a) = \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a)\, v_{\pi_k}(s')$, Monte Carlo 直接从原始 $q_{\pi_k}(s,a)$ 的定义式出发，直接估计 action value $q_{\pi_k}(s,a)$

MC method aims to solve
$$
q_\pi(s,a)=\mathbb{E}\!\left[\,R_{t+1}+\gamma R_{t+2}+\cdots \mid S_t=s,\ A_t=a\,\right],\quad \forall\, s,a.
$$
其实就是用一个 policy $\pi$ 生成一个 episode，然后用 episode 的 return 去直接近似估计 action value，第一个 episode 进来 $q(s,a) \approx r_{t+1} + \gamma r_{t+2} + \cdots$，然后后面的 episode 不断进来，增量式更新期望的近似估计



**进行估计的时候如何更加充分地利用数据**
一次采样得到 original episode，可以在后面截取出多种 episode
$$
\begin{aligned}
& s_1 \xrightarrow{a_2} s_2 
   \xrightarrow{a_4} s_1 
   \xrightarrow{a_2} s_2 
   \xrightarrow{a_3} s_5 
   \xrightarrow{a_1} \cdots
   \quad\text{[original episode]} \\[6pt]
& s_2 \xrightarrow{a_4} s_1
   \xrightarrow{a_2} s_2
   \xrightarrow{a_3} s_5
   \xrightarrow{a_1} \cdots
   \quad\text{[episode starting from $(s_2, a_4)$]} \\[6pt]
& s_1 \xrightarrow{a_2} s_2
   \xrightarrow{a_3} s_5
   \xrightarrow{a_1} \cdots
   \quad\text{[episode starting from $(s_1, a_2)$]} \\[6pt]
& s_2 \xrightarrow{a_3} s_5
   \xrightarrow{a_1} \cdots
   \quad\text{[episode starting from $(s_2, a_3)$]} \\[6pt]
& s_5 \xrightarrow{a_1} \cdots
   \quad\text{[episode starting from $(s_5, a_1)$]}
\end{aligned}
$$
first visit，一个 episode 中，只计入某个 $(s,a)$ 第一次起始的链条
every-visit，一个 episode 中，计入某个 $(s,a)$ 的所有子起始链条 

###### incremental update
正常 Monte Carlo 估计，每个 acition value 要用多个 episode 取均值去估计，然后再去计算 $q_{\pi_k}(s,a)$ ，再去迭代
更高效的方法是，得到一个 episode 直接就去估计  $q_{\pi_k}(s,a)$，拿到下一个 episode 再结合之前的 episode 重新估计出 $v_{\pi_k}(s)$ 

###### Generalized policy iteration 框架
policy evaluation和 policy improvement 不断切换，每一步 Bellman 公式去迭代求解 $v_{\pi_k}(s)$  , 可能没有收敛就跳到 policy improvement, 但是没关系, 即使评估是近似的，只要 $v_{\pi_k}(s)$ 的估计足够好，基于 $v_{\pi_k}(s)$ 做贪婪改进通常会产生一个更优的策略。

###### soft policy
exploring starts 指的是: 为了保证所有的 $(s,a)$ 都能被遍历，比较笨的办法就是让所有每种 $(s,a)$ 都作为 start 来采样 episode，但是实际应用中这种做法可能非常麻烦，比如一个机器人在一个场地里，所有开始点我都得搬过去
采用 stochastic 的 policy，这样只要 episode 足够长，所有的 $(s,a)$ 就都能被访问到，这样就可以避免 exploring starts

######  $\epsilon$-greedy
$\epsilon$-greedy policy就是一种 soft policy，就是 s 处所有 a 挑一个概率最大，剩下的 a 均分概率，目的是扩 大搜索空间，$\epsilon \in [0,1]$ 取值为 0 就是 deterministic, 取值为1就是均匀分布，从0变到1，探索性越来越强，最优性越来越差 ；
$\epsilon$-greedy policy 往往 episode 很长，会用 **every-visit**

**consistent.** 如果 $\epsilon$-greedy policy 每一步的概率最大的策略和 $\epsilon=0$ 对应的最优策略是一致的，那么就称为和最优是 consistent的；增大 $\epsilon$ 的过程要尽量保证 consistent

核心思想和就是: **Balance between exploitation and exploration** , 如果过分贪婪到 deterministic，虽然充分利用，最优性最好，但是探索区域变少了，信息可能不完备


# Stochastic approximation

神经网络就是一个函数 $g(w)$ ,我们如果要求解方程 $g(w)=0$ ,
要么有模型(知道 $g$ 表达式)，要么有数据；

**Stochastic approximation 的本质是：用带噪声样本迭代去解一个“期望/不动点/根方程”。**

#### mean estimation
不等待整个批次 n 样本进来之后再估计，迭代式更新期望估计
实际中，calculate the mean $\bar{x}$ incrementally，进来一个数据更新一次均值，初始 $w1=x1$ 
$$
w_{k+1} = w_k - \frac{1}{k}\left(w_k - x_k\right).
$$
$\{w_k\}$ 会收敛至期望 $E[X]$

#### 更一般的情况
上面的系数 $\frac{1}{k}$ 换成 $\alpha_k$ , 逻辑上不是严格的平均值增量更新
$$
w_{k+1} = w_k - \alpha_k \left(w_k - x_k\right).
$$
**这个其实不仅是 mean estimation 的一般情况，也是 EMA 的一般情况**，EMA (exponential moving average, 具体推导见 RQ-VAE 笔记) 公式为 $m_t=(1-\alpha)m_{t-1}+\alpha x_t$，其中的系数 $\alpha$ 是定值

## Robbins-Monro algorithm

mean estimation 和随机梯度下降都是RM算法的特殊情况

问题还是**求解 $g(w)=0$ 的根 $w^*$**,  具体 $g$ 的表达式不知道

$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k), \qquad k = 1,2,3,\ldots$$

本质上就是没有模型就要有数据，按理有一个 $w_k$ , 代入即可得 $g(w_k)$ , 但我们不知道 $g$ 表达式，没法这么做，只能够**观测得到带有噪声的 $\tilde{g}(w_k, \eta_k)$ ，要求 $\tilde{g}(w_k, \eta_k)$ 在精度要求内能够近似  $g(w_k)$** 

**Theorem (Robbins–Monro Theorem)**
In the Robbins–Monro algorithm, if
1) $0 < c_1 \le \nabla_w g(w) \le c_2$ for all $w$;  # $w$ 如果是向量，上下界就是矩阵
2) $\sum_{k=1}^{\infty} a_k = \infty$ and $\sum_{k=1}^{\infty} a_k^2 < \infty$;  # 保证$a_k$ 收敛到0，但是收敛到0的速度不那么快，比如$\frac{1}{k}$
3) $\mathbb{E}[\eta_k \mid \mathcal{H}_k] = 0$ and $\mathbb{E}[\eta_k^2 \mid \mathcal{H}_k] < \infty$;
where $\mathcal{H}_k = \{w_k, w_{k-1}, \ldots\}$, then $w_k$ converges with probability $1$ to the root $w^*$ satisfying $g(w^*) = 0$.

#### mean estimation 为什么是 RM 的特殊情况
目标是拟合期望，令 $g(w)=w-E[X]$ , 就是解这个方程，因为 $\tilde{g}(w_k, \eta_k)$ 是近似 $g(w)$ 的，取 $\alpha_k=\frac{1}{k},\quad \tilde{g}(w_k, \eta_k)=w-x$，因为要验证噪声项 $\eta_k$，代入噪声项定义式即可得:
$$\eta_k \coloneqq g-\tilde{g}=x-E[X]$$


## SGD
mean estimation 是 SGD 的一个特殊情况
SGD 是 RM 算法的一个特殊情况

SGD 就是批次梯度下降批次数取 1 的特殊情况：
$\min_{w}J(w) = \mathbb{E}[f(w,X)] = \mathbb{E}\left[\frac{1}{2}\left\lVert w - X\right\rVert^{2}\right]$
在这个目标函数下，求梯度，批次大小为1，即随机采样一个样本进行梯度下降的估计可得:
$w_{k+1} = w_k - \alpha_k \nabla_w f(w, x_k) = w_k - \alpha_k (w_k - x_k)$
其中 $\alpha_k$ 取 $\frac{1}{k}$ 即为 mean estimation

###### 证明 SGD 是一个特殊的 RM 算法，从而证明 SGD 的收敛性  
SGD 求解的问题是 $\min_{w}J(w) = \mathbb{E}[f(w,X)]$，RM 算法求解的问题是 $g(w)=0$ ，
构造 $g(w) = \nabla_w J(w) = \mathbb{E}\!\left[\nabla_w f(w, X)\right]$，
SGD 中一次采样就是一次带有噪声的观测，即:
$$
\tilde{g}(w,\eta)=\nabla_w f(w,x)
$$
其中噪声 $\eta \coloneqq g-\tilde{g}=\nabla_w f(w,x)\mathbb-{E}\!\left[\nabla_w f(w,X)\right]$
从而 $w_{k+1}=w_k-a_k,\tilde{g}(w_k,\eta_k)=w_k-a_k,\nabla_w f(w_k,x_k)$ ，得证 SGD 就是特殊的 RM 算法

###### SGD 的性质
The relative error between the stochastic and true gradients is

$$
\delta_k \doteq \frac{\left\|\nabla_w f(w_k, x_k) - \mathbb{E}\!\left[\nabla_w f(w_k, X)\right]\right\|}{\left\|\mathbb{E}\!\left[\nabla_w f(w_k, X)\right]\right\|}.
$$

满足一定条件下，有

$$\delta_k \le \frac{\left\|\nabla_w f(w_k,x_k)-\mathbb{E}\!\left[\nabla_w f(w_k,X)\right]\right\|}{c\left\|w_k-w^{*}\right\|}$$

这个上界是一个变化量；$w_k$ 距离 $w_*$ 很远的时候，相对误差 $\delta_k$ 上界小，$w_k$ 距离 $w_*$ 很近的时候，相对误差 $\delta_k$ 上界大；这保证了 SGD 在距离最优解远的时候，不会"乱跑"，基本上朝着最优解稳定前进，距离最优解很近的时候，才会呈现较大的随机性；


###### BGD MBGD SGD 对比
优化问题为 $\min_{w}J(w) = \mathbb{E}[f(w,X)]$ , 那么标准的 GD 的形式就是
$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)
$$

涉及随机变量，我们没办法直接求理论值，也没法求梯度；所以用经验风险去拟合: $J_n(w)=\frac{1}{n}\sum_{i=1}^{n} f(w,X_i)$, 具体用多大的批次去拟合这个，得到了下面三种:
BGD  
$$
w_{k+1}=w_k-\alpha_k\frac{1}{n}\sum_{i=1}^{n}\nabla_w f(w_k,x_i)
$$
mini-batch GD  
$$
w_{k+1}=w_k-\alpha_k\frac{1}{m}\sum_{j\in \mathcal{I}_k}\nabla_w f(w_k,x_j)
$$
SGD
$$
w_{k+1}=w_k-\alpha_k\nabla_w f(w_k,x_k)
$$

说白了一句话总结就是，$\min_{w}J(w)$ ，GD 要用到梯度，涉及随机变量没法直接算，那么用多少个样本去具体估计这个梯度；



---



# Temporal-Difference Learning

强化学习的目的是找最优策略
#### TD learning algorithm 具体形式：
策略 $\pi$ 下产生了 $\{s_t, r_{t+1}, s_{t+1}\}$ ，进行如下迭代
$$
v_{t+1}(s_t)
= v_t(s_t) - \alpha_t(s_t)\Big( v_t(s_t) - \big[r_{t+1} + \gamma v_t(s_{t+1})\big] \Big)
\tag{1}
$$
$$
v_{t+1}(s)
= v_t(s), \qquad \forall\, s \neq s_t
\tag{2}
$$


$r_{t+1} + \gamma v_t(s_{t+1})$ 称为 TD target，这玩意在状态 $s_t$ 下去期望就是 state value 的定义
$v_t(s_t) - \big[r_{t+1} + \gamma v_t(s_{t+1})\big]$ 称为 TD error 

TD learning algorithm 做的是一个固定策略 $\pi$ 下的 policy evaluation, 是一个 model free 的单步 bootstrap 迭代，**本质上是在没有模型的情况下使用 RM 算法求解 Bellman 公式** (之前介绍的 Bellman 公式的直接迭代需要依赖具体模型)
Monte Carlo 虽然也是 model free，但是其直接估计的是期望式子；

#### TD target
$v_{t+1}(s_t)= v_t(s_t) - \alpha_t(s_t)\Big( v_t(s_t) - \big[r_{t+1} + \gamma v_t(s_{t+1})\big] \Big)\tag{1}$ 中 $r_{t+1} + \gamma v_t(s_{t+1})$ 叫做 target，其实就是在做一个很简单的事情，就是把当前的估计目标值上拉，只不过这个目标值往往来自与环境交互的采样；比如这里当前估计值是 $v_t(s_t)$ , 如果跟 target 这次的采样的 target 相等，就不更新，如果 $v_t(s_t)  - target = k > 0$ , 更新出来的 $v_{t+1}(s_t)$ 显然有  $v_{t+1}(s_t) - target$ 更小，也就实现了往目标值拉进的效果； 

#### n-step Sarsa

Sarsa 算法的 policy evaluation 和 policy improvement 过程见 p135

---


![[Pasted image 20260304162352.png]]


---


$$
\begin{aligned}
G_t^{(1)} &= R_{t+1} + \gamma\, q_{\pi}(S_{t+1}, A_{t+1}) 
&& \rightarrow\ \text{Sarsa}\\
G_t^{(2)} &= R_{t+1} + \gamma R_{t+2} + \gamma^{2}\, q_{\pi}(S_{t+2}, A_{t+2})
&& \\
&\vdots\\
G_t^{(n)} &= R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n}\, q_{\pi}(S_{t+n}, A_{t+n})
&& \rightarrow\ n\text{-step Sarsa}\\
&\vdots\\
G_t^{(\infty)} &= R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \gamma^{3} R_{t+4} + \cdots
&& \rightarrow\ \text{MC}
\end{aligned}
$$
$q_{\pi}(S_{t+1}, A_{t+1})$ 中的大写 $S_{t+1}, A_{t+1}$ 表示随机变量的某次采样，只是还没采样得到具体数值，但是代表的是某个采样的数值

discounted return** is: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
这里 $G_t^{(1)} \neq G_t$  , 二者在条件期望的意义下相等(该结论对 1-step 到 $\infty$-step 都成立)，证明如下：
$$q_{\pi}(s,a)=\mathbb{E}_{\pi}\!\left[\,R_{t+1}+\gamma G_{t+1}\mid S_t=s,\;A_t=a\,\right]$$
$$
\mathbb{E}_{\pi}\!\left[\,G_{t+1}\mid S_t=s,\;A_t=a\,\right]
=
\mathbb{E}_{\pi}\!\left[\,
\mathbb{E}_{\pi}\!\left[\,G_{t+1}\mid S_{t+1},\;A_{t+1}\,\right]
\mid S_t=s,\;A_t=a
\,\right]
$$
无条件也是一种条件，重期望原始公式为 $\mathbb{E}(X)=\mathbb{E}\big(\mathbb{E}(X\mid Y)\big)$ , 两边的 E 括号里面都是随机变量，两边括号里面各自加上相同的一个条件自然等式仍然成立


---


#### TD algorithm 总结对比

#### 本质上只是 TD-target 不同
All the algorithms we introduced in this lecture **can be expressed in a unified expression:**

$$
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
-
\alpha_t(s_t,a_t)\Bigl[q_t(s_t,a_t)-\bar q_t\Bigr],
$$

where $\bar q_t$ is the **TD target**.

Different TD algorithms have different $\bar q_t$.
**Sarsa：**
$$
\bar q_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}).
$$

**$n$-step Sarsa：**
$$
\bar q_t = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_t(s_{t+n}, a_{t+n}).
$$

**Expected Sarsa：**
$$
\bar q_t = r_{t+1} + \gamma \sum_a \pi_t(a\mid s_{t+1})\, q_t(s_{t+1}, a).
$$

**Q-learning：**
$$
\bar q_t = r_{t+1} + \gamma \max_a q_t(s_{t+1}, a).
$$

**Monte Carlo：**
$$
\bar q_t = r_{t+1} + \gamma r_{t+2} + \cdots .
$$
MC 方法也可以写进这个统一表达式：令$\alpha_t(s_t,a_t)=1$ ， 则有 $q_{t+1}(s_t,a_t)=\bar q_t$



---


#### 本质上都在解相应 Bellman 方程

Bellman 视角：这些算法都是 stochastic approximation

All the algorithms can be viewed as **stochastic approximation algorithms** solving the **Bellman equation** or **Bellman optimality equation**：

**Sarsa（BE）：**
$$
q_\pi(s,a)
=
\mathbb{E}\!\left[
R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,\ A_t=a
\right].
$$

**$n$-step Sarsa（BE）：**
$$
q_\pi(s,a)
=
\mathbb{E}\!\left[
R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^n q_\pi(s_{t+n},a_{t+n})
\mid S_t=s,\ A_t=a
\right].
$$

**Expected Sarsa（BE）：**
$$
q_\pi(s,a)
=
\mathbb{E}\!\left[
R_{t+1}+\gamma\,\mathbb{E}_{A_{t+1}}\!\left[q_\pi(S_{t+1},A_{t+1})\right]
\mid S_t=s,\ A_t=a
\right].
$$

**Q-learning（BOE）：**
$$
q(s,a)
=
\mathbb{E}\!\left[
R_{t+1}+\max_a q(S_{t+1},a)
\mid S_t=s,\ A_t=a
\right].
$$

**Monte Carlo（BE）：**
$$
q_\pi(s,a)
=
\mathbb{E}\!\left[
R_{t+1}+\gamma R_{t+2}+\cdots
\mid S_t=s,\ A_t=a
\right].
$$



---



#### on-policy 和 off-policy

behavior policy $\beta$ ：负责采样数据的 policy
target policy $\pi$ ：我们想要改进的 policy

on-policy: behavior policy 和 target policy 相同；即用 $\pi$ 采样的数据学习 $\pi$
off-policy: behavior policy 和 target policy 不同；比如用一个 exploration 很强的 policy 产生的数据来改进目标 policy

直觉理解： on-policy 就是自己玩游戏，自己摸索进步；off-policy 就是看别人咋玩的，然后改进自己的玩法；

#### on-policy 和 off-policy 版 Q-learning

Q-learning on-policy：后面接一个 $\epsilon$-greedy 的 policy improvement 的过程；就是和 *Algorithm 7.1:  Sarsa 算法完整版核心(p135)* 后半部分是一样的；on-policy 必须要有后面的这个 policy improvement，因为他要自采数据，每步更新 policy 之后再采集数据
Q-learning off-policy：采数据是另一个 policy 的事情，更新的策略就不用有 exploration , 因此后面就是纯 greedy 的 action 选择；并且由于数据是提前别的策略就搞好的，新更新的策略产生的数据也不用于自己的训练；同时，off-policy 也可以选择不断迭代更新 action value，只在最后一步进行纯 greedy 的 action 选择来得到最终的 policy (反正不用自己产生数据，中间不得到策略也行)




---




# Value function approximation

###### Objective function 
$$
J(w) = \mathbb{E}\!\left[\big(v_{\pi}(S) - \hat{v}(S,w)\big)^2\right].
$$
其中的 $S$ 是状态空间中的随机变量，$w$ 为函数参数，自然就有概率分布，就能求期望.


###### Stationary distribution
要求上面 Objective function 这个期望式子，自然要知道随机变量 $S$ 的概率分布，均匀分布是一种想法，但是这样就没有侧重点；

策略 $\pi$ 下走一个非常长的 episode，最终每个状态到达的次数占总次数的比例趋于稳定，称之为 **stationary distribution，描述是趋于稳定时，agent 处于每个状态的概率**，记作 $d_{\pi}(S)$ ,  则 
$$
J(w)=\sum_{s\in \mathcal{S}} d_{\pi}(s)\,\big(v_{\pi}(s)-\hat{v}(s,w)\big)^2
$$
上面这个式子中的 $\hat{v}(s,w)\big)$ 不涉及策略，是对给定策略下的 v 的拟合(往下看 Deep Q learning 的内容) 
事实上知道概率转移矩阵$P_{\pi}$ ，*$P_\pi(i,j)$ 为 $s_i$ 向 $s_j$ 的转移概率，每一行元素和为1* , 能直接预测出 $d_{\pi}$
$$
d_{\pi}^{\top} = d_{\pi}^{\top} P_{\pi}.
$$
因为，定义第 $t$ 步状态的分布 $\mu_t(s) = P(S_t = s)$ ，有$\mu_t^{\top} = [\mu_t(1), \ldots, \mu_t(|S|)]$，又
$$
\mu_{t+1}(s') = \sum_{s} \mu_t(s)\, P_{\pi}(s, s').
$$
写成矩阵形式为：
$$
\mu_{t+1}^{\top} = \mu_t^{\top} P_{\pi}.
$$
*(如果写成列向量形式就出现 $P_{\pi}^{\top}$ ，保留原始转移矩阵形式，就写成上式的行向量形式)*


###### 参数更新

就是对目标函数进行梯度下降

$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)
$$

The true gradient is

$$
\begin{aligned}
\nabla_w J(w)
&= \nabla_w \mathbb{E}\!\left[\left(v_{\pi}(S)-\hat{v}(S,w)\right)^2\right] \\
&= \mathbb{E}\!\left[\nabla_w \left(v_{\pi}(S)-\hat{v}(S,w)\right)^2\right] \\
&= 2\,\mathbb{E}\!\left[\left(v_{\pi}(S)-\hat{v}(S,w)\right)\left(-\nabla_w \hat{v}(S,w)\right)\right] \\
&= -2\,\mathbb{E}\!\left[\left(v_{\pi}(S)-\hat{v}(S,w)\right)\nabla_w \hat{v}(S,w)\right].
\end{aligned}
$$

比如网格任务中，网格中状态用平面坐标 $(x,y)$ 表示， 
比如假设  $\phi(s) = [1, x, y, x^2, y^2, xy]^{\top}$, $\hat{v}(s,w) = \phi^{\top}(s)w= w_1 + w_2 x + w_3 y + w_4 x^2 + w_5 y^2 + w_6 xy$
而 $v_{\pi}(S)$ 是不知道的，可以**用 MC 方法采样来近似** 或者 **用TD-target 近似**
这样一来上面梯度更新的式子就能进行了


## Deep Q-learning (Deep Q Network)

Tabular 的方法没法处理连续空间

DQL 这种东西因为涉及 $E(...S,...,A,...)$ , 总之就是 E 里面有两个随机变量 S 和 A ，那么自然要考虑他们俩的联合概率分布才能去处理期望；无信息先验就假设他们俩为二维均匀分布，有了这个分布假设，所以需要 experience replay 的方式去用数据迭代；

Tabuler 的方法不引入这种二维均匀的先验假设，就可以直接按照采样顺序利用数据；但是 Tabular 的方法如果是 off-line 的，也可以用 experience repaly (采集之后，每次用随机从里面抽一个$(s,a)$ )，因为 off-line 本身数据就不是目标策略产生的，不按时序输进来的 experience replay 自然也是适用的，并且更高效，因为这样同一个 sample 可以用多次

DQL 俩 network
- One is a **main network** representing $\hat{q}(s,a,w)$.
- The other is a **target network** $\hat{q}(s,a,w_T)$.
The objective function in this case degenerates to
$$
J=\mathbb{E}\!\left[\left(R+\gamma\max_{a\in\mathcal{A}(S')}\hat{q}(S',a,w_T)-\hat{q}(S,A,w)\right)^2\right].
$$
其中 main network 通过训练更新参数 $w$ , target network 结构上跟 main network 一样，target network 不用训练，隔一段时间把 main network 中参数赋值过来就行；
实际执行流程：
1. 从 replay buffer $B$  **mini-batch**
2. target net（参数 $w_T$）每 C 个 mini-batch 复制一次 main network 中的参数
3. mini-batch 更新 main net 参数 $w$

这里是直接逼近最优价值函数


---



# Policy Gradient

## value-based 与 policy-based

这部分前面所有的笔记的方法都是 value-based，都是得到 action value，然后 $\epsilon$-greedy 选策略

## basic idea

之前都是 tabular 形式表示 policy；改用含参函数，常见的记号有: $\pi(a \mid s,\theta)$ , $\pi_{\theta}(a \mid s)$

tabular 形式下最优策略的定义为：$v_{\pi^*}(s) \ge v_{\pi}(s), \quad\forall \pi, \ \forall s \in S$；查策略的时候直接查表就能得到；
用函数定义的 optimal policy，因为是类似得到最大的 state value 这种东西，所以通过最大化某个标量目标函数 $J(\theta)$ 来得到，更新策略用梯度上升 ；查策略的时候，把 s 输入神经网络中前向传播一次得到 $\pi(a \mid s,\theta)$


## metrics

下面介绍的这些 metrics， maximize 他们然后更新参数；
实际上 $\bar r_\pi \;=\; (1-\gamma)\,\bar v_\pi$ ，因此下面这几种本质上是等价的

###### average state value
分布 $d(s)$ 加权 state value, state value 本身就是策略 $\pi$ 的函数；分布 $d(s)$ 可能与 $\pi$ 有关，可也能无关；比如可以选 stationary distribution 作为 $d(s)$ 
$$
\bar v_\pi \;=\; \sum_{s\in\mathcal{S}} d(s)\, v_\pi(s).
$$

###### average state value 另一种等价定义 (论文中会见到)

$$
J(\theta)\;=\;\mathbb{E}\!\left[\sum_{t=0}^{\infty}\gamma^{t}R_{t+1}\right].
$$
证明：
$$
J(\theta)
=\mathbb{E}\!\left[\sum_{t=0}^{\infty}\gamma^{t}R_{t+1}\right]
=\sum_{s\in\mathcal{S}} d(s)\,\mathbb{E}\!\left[\sum_{t=0}^{\infty}\gamma^{t}R_{t+1}\,\Big|\, S_0=s\right]
=\sum_{s\in\mathcal{S}} d(s)\,v_\pi(s)
=\bar v_\pi.
$$
本质上就是积分求和换序，两个随机变量 $(S,A)$，对二维区域积分，先把 $S$ 积出来

###### average one-step reward
$$
\bar r_\pi \;:=\; \sum_{s\in\mathcal{S}} d_\pi(s)\, r_\pi(s)
\;=\; \mathbb{E}\bigl[r_\pi(S)\bigr],
$$
where $S \sim d_\pi$. Here,
$$
r_\pi(s) \;:=\; \sum_{a\in\mathcal{A}} \pi(a\mid s)\, r(s,a). 
$$
$$
r(s,a)\;:=\;\mathbb{E}\!\left[\,R\mid s,a\,\right]
\;=\;\sum_{r} r\,p(r\mid s,a).
$$
这里 $r_\pi(s)$ 和 $r(s,a)$ 的计算都涉及加权，是因为某个状态下选动作可以是非确定性的随机分布，某个状态选了某个动作获得的 reward 也可以是非确定性的随机分布


###### average one-step reward 另一种等价定义 (论文中会见到)

Suppose an agent follows a given policy and generate a trajectory with the rewards as $(R_{t+1}, R_{t+2}, \ldots)$.

The average single-step reward along this trajectory is
$$
\lim_{n\to\infty}\frac{1}{n}\,\mathbb{E}\!\left[\,R_{t+1}+R_{t+2}+\cdots+R_{t+n}\mid S_t=s_0\,\right]
=
\lim_{n\to\infty}\frac{1}{n}\,\mathbb{E}\!\left[\,\sum_{k=1}^{n} R_{t+k}\mid S_t=s_0\,\right].
$$
where $s_0$ is the starting state of the trajectory.

$$
\lim_{n\to\infty}\frac{1}{n}\,\mathbb{E}\!\left[\,\sum_{k=1}^{n} R_{t+k}\mid S_t=s_0\,\right]
=
\lim_{n\to\infty}\frac{1}{n}\,\mathbb{E}\!\left[\,\sum_{k=1}^{n} R_{t+k}\,\right]
=
\sum_{s} d_\pi(s)\, r_\pi(s)
=
\bar r_\pi.
$$
无穷步的时候，起点在哪不重要了，所以上面的式子可以直接去掉条件 $S_t=s_0$


## gradients of the metrics

说白了 state value 是 $\pi$ 的函数，从原始的 $J(\theta)$ 梯度传到函数形式上的 $\pi(a\mid s,\theta)$

$$
J(\theta)=\sum_{s}\mu(s)\,v_{\pi}(s).
$$

$$
\nabla_{\theta}J(\theta)=\sum_{s}\mu(s)\,\nabla_{\theta}v_{\pi}(s).
$$

$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)\,q_{\pi}(s,a).
$$

$$
\nabla_{\theta}J(\theta)
=\sum_{s}\eta(s)\sum_{a}\nabla_{\theta}\pi(a\mid s,\theta)\,q_{\pi}(s,a).
$$

实际我们会写成下面等价的期望形式，因为期望形式我们可以 MC 采样
$$
\nabla_{\theta}J(\theta)
=\mathbb{E}_{S\sim d_{\pi},\,A\sim \pi}\!\left[\nabla_{\theta}\ln \pi(A\mid S,\theta)\,q_{\pi}(S,A)\right].
$$
严格来说上面 $(S,A)$ 服从的是二维分布
证明如下：
 $$
\nabla_{\theta}\ln\pi(a\mid s,\theta)
=\frac{\nabla_{\theta}\pi(a\mid s,\theta)}{\pi(a\mid s,\theta)}.
$$

$$
\nabla_{\theta}\pi(a\mid s,\theta)
=\pi(a\mid s,\theta)\,\nabla_{\theta}\ln\pi(a\mid s,\theta).
$$
把 $\nabla_{\theta}\pi(a\mid s,\theta)$ 代入 $\nabla_{\theta}J(\theta)=\sum_{s}\eta(s)\sum_{a}\nabla_{\theta}\pi(a\mid s,\theta)\,q_{\pi}(s,a)$ 即可

实际由于用神经网络拟合，可能出现负值，最后 softmax 概率化一下作为 $\pi(a\mid s,\theta)$ 然后再进 $\ln \pi(A\mid S,\theta)$；$h$ 是神经网络
$$
\pi(a\mid s,\theta)
=\frac{e^{h(s,a,\theta)}}{\sum_{a'\in\mathcal{A}} e^{h(s,a',\theta)}}.
$$

但注意这种概率化 $\pi(a\mid s,\theta)$ 要求空间 A 是有限的，如果有无数种可能的 action，那么就没法 softmax 了，如果有无数种 action，那么就要求选用 deterministic policy


#### REINFORCE

**Initialization:** A parameterized function $\pi(a\mid s,\theta)$, $\gamma\in(0,1)$, and $\alpha>0$.  
**Aim:** Search for an optimal policy maximizing $J(\theta)$.

For the $k$-th iteration, do

- Select $s_0$ and generate an episode following $\pi(\theta_k)$. Suppose the episode is
  $\{s_0,a_0,r_1,\ldots,s_{T-1},a_{T-1},r_T\}$.

- For $t=0,1,\ldots,T-1$, do
     **Value update:** $q_t(s_t,a_t)=\sum_{k=t+1}^{T}\gamma^{\,k-t-1}r_k$  
     **Policy update:** $\theta_{t+1}=\theta_t+\alpha\nabla_{\theta}\ln\pi(a_t\mid s_t,\theta_t)\,q_t(s_t,a_t)$
  $\theta_k=\theta_T$ (一长串 episode 不同起点位置得到的很多个数据全部迭代完了之后，最后再赋给 $\theta_k$ 用于产生下一次的 episode；这样一个 episode 分出的 $t=0,1,\ldots,T-1$ 个对应起点的数据是同一个 $\pi(\theta_k)$ 产生的；


# Actor-critic Method

“actor” refers to **policy update**. It is called *actor* is because the policies will be applied to take actions. 说白了 actor 就是管策略相关的东西

“critic” refers to **policy evaluation** or **value estimation**. It is called *critic* because it criticizes the policy by evaluating it. 说白了 critic 就是管如何评估某个策略，比如用 action value 之类的东西

###### 最简单的一种 AC，QAC 算法(Q-Actor-Critic)
QAC 的 Q 指的就是 action value 的符号表示 $q$

Search for an optimal policy by maximizing $J(\theta)$.

At time step $t$ in each episode, do

Generate $a_t$ following $\pi(a\mid s_t,\theta_t)$, observe $r_{t+1}, s_{t+1}$, and then generate $a_{t+1}$ following $\pi(a\mid s_{t+1},\theta_t)$.  说白了就是拿到 Sarsa 所需的 $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$ 

**Critic (value update):** (其实就是 Sarsa 结合 value function approximation)
critic 学的是 action value 或者 state value， 其参数 $w$ 的更新基于最小化$\bigl(y_T-\hat{q}(s,a,w)\bigr)^2$ 把原本需要 main net 和 target net 延迟更新的 $y_T$ 换成 TD-target，然后梯度下降；
$$
w_{t+1}
=
w_t
+
\alpha_w\Bigl[
r_{t+1}
+\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
\Bigr]\nabla_w q(s_t,a_t,w_t)
$$


**Actor (policy update):** 
actor 学的是策略 $\pi$ , 其参数 $\theta$ 以 $J(\theta)$ 为目标函数梯度上升：
$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta \nabla_\theta \ln \pi(a_t\mid s_t,\theta_t)\,q(s_t,a_t,w_{t+1})
$$


其实就是 q 和 $\pi$ 都用 value function approximation 的方法，都引入各自的参数。比如分别用神经网络去拟合这俩东西；还有 $\epsilon$-greedy 本质上是引入 stochastic 从而增加 exploration，上面的 value function approximation 的 $\pi(a_t \mid s_t,\theta_t)$ 本身就带有随机性；


###### Advantage Actor Critic (两个A，所以缩写为 A2C)
引入一个偏置项来减少估计的方差
**Property:** the policy gradient is invariant to an additional baseline:
$$
\nabla_{\theta}J(\theta)
=\mathbb{E}_{S\sim\eta,\,A\sim\pi}\!\left[\nabla_{\theta}\ln\pi(A\mid S,\theta)\,q_{\pi}(S,A)\right]
$$
$$
=\mathbb{E}_{S\sim\eta,\,A\sim\pi}\!\left[\nabla_{\theta}\ln\pi(A\mid S,\theta)\,\bigl(q_{\pi}(S,A)-b(S)\bigr)\right]
$$
证明见第十课第二个视频(3:03)，其实就是写开来，b(s)放到累次求和前面，第二个求和符合和求导可以换序，$\Sigma \pi=1$ ，常数求导为 0   

因此上面加了偏置 $b(S)$ 之后期望不变，但是方差会改变 : 
$$
X(S,A)=\nabla_{\theta}\ln \pi(A\mid S,\theta)\,\bigl[q(S,A)-b(S)\bigr].
$$
首先 X 会是一个向量，因为参数 $\theta$ 往往是向量；
对于 X 的协方差矩阵，本身我们无法比较矩阵的大小；常见的方法是比较矩阵的 F 范数或者 trace(协方差矩阵的 trace 恰好就是各个分量的方差和)

$$
\operatorname{Var}(X))
=(\mathbb{E}[XX^{\top}]-\mathbb{E}[X]\mathbb{E}[X]^{\top}\Big).
$$
由 trace 的轮换不变得
$$
\operatorname{tr}[\operatorname{Var}(X)]
=\mathbb{E}\!\left[X^{\top}X\right]-\bar{x}^{\top}\bar{x}
\qquad (\bar{x}=\mathbb{E}[X]).
$$
trace 和 期望可换序
$$
\operatorname{tr}\!\big(\mathbb{E}[XX^{\top}]\big)
=\mathbb{E}\!\big[\operatorname{tr}(XX^{\top})\big].
$$
因此： 
$$
\begin{aligned}
\mathbb{E}\!\left[X^{\top}X\right]
&=\mathbb{E}\!\left[\left((\nabla_{\theta}\ln\pi)^{\top}(\nabla_{\theta}\ln\pi)\,\bigl(q(S,A)-b(S)\bigr)\right)^{2}\right] \\
&=\mathbb{E}\!\left[\|\nabla_{\theta}\ln\pi\|^{2}\,\bigl(q(S,A)-b(S)\bigr)^{2}\right].
\end{aligned}
$$

所以选取不同的 $b(S)$ 会影响方差，目标是方差最小，理论上可以证明最优结果为：
$$
b^{*}(s)
=\frac{\mathbb{E}_{A\sim\pi}\!\left[\left\|\nabla_{\theta}\ln\pi\!\left(A\mid s,\theta_t\right)\right\|^{2}\,q(s,A)\right]}
{\mathbb{E}_{A\sim\pi}\!\left[\left\|\nabla_{\theta}\ln\pi\!\left(A\mid s,\theta_t\right)\right\|^{2}\right]}.
$$

但是这个式子太复杂，实际直接用下面的结果近似：
$$
b(s)=\mathbb{E}_{A\sim\pi}\!\left[q(s,A)\right]=v_{\pi}(s).
$$

选用 $b(s)=v_{\pi}(s)$ 加入偏置项之后参数的更新为：
$$
\begin{aligned}
\theta_{t+1}
&=\theta_t+\alpha\,\nabla_{\theta}\ln\pi(a_t\mid s_t,\theta_t)\,\delta_t(s_t,a_t)\\
&=\theta_t+\alpha\,\frac{\nabla_{\theta}\pi(a_t\mid s_t,\theta_t)}{\pi(a_t\mid s_t,\theta_t)}\,\delta_t(s_t,a_t)\\
&=\theta_t+\alpha\left(\frac{\delta_t(s_t,a_t)}{\pi(a_t\mid s_t,\theta_t)}\right)\nabla_{\theta}\pi(a_t\mid s_t,\theta_t).
\end{aligned}
$$
其中
$$
\delta_t
= q_t(s_t,a_t)-v_t(s_t)
$$
叫做 **advantage function，其实就是关注相对值**，你一个状态 action value 绝对值没有意义，只有相对于其他状态 action value 更大或者更小才有意义，因此这里减去 $v_t(s_t)$ ，而 $v_t(s_t)$ 就是 $q_t(s_t,a_t)$ 的期望，因此说白了就是一个均值归一化；

$$
\delta_t
= q_t(s_t,a_t)-v_t(s_t)
\;\to\;
r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t).
$$
**也可以用上面的 TD-target 替换，换了之后就是 TD error，能够替换的原因是他们在期望意义下相等**：
$$
\mathbb{E}\!\left[q_{\pi}(S,A)-v_{\pi}(S)\mid S=s_t,\;A=a_t\right]
=
\mathbb{E}\!\left[R+\gamma v_{\pi}(S')-v_{\pi}(S)\mid S=s_t,\;A=a_t\right].
$$
符号 $S^\prime$ 跟 $S_{t+1}$ 一个意思；换成 TD-target 之后，我们只用一个神经网络近似 $v_t(s)$ 就行，如果用原始的 $q_t(s_t,a_t)-v_t(s_t)$ ， 还要用两个神经网络分别近似 $q,v$



###### Advantage actor-critic (A2C) or TD actor-critic

从上面引入偏置项之后得到更关注相对值的 $\delta_t$ ，然后换成 TD-target 之后的形式，类比前面的 QAC 给出下面的算法

**Aim:** Search for an optimal policy by maximizing $J(\theta)$.

At time step $t$ in each episode, do

Generate $a_t$ following $\pi(a_t\mid s_t,\theta_t)$ and then observe $r_{t+1},\, s_{t+1}$.

TD error (advantage function):
$$
\delta_t \;=\; r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)
$$

Critic (value update):
以 $L_t(w)=\frac{1}{2}\Big(r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w)\Big)^2$ 为目标的梯度下降

$$
w_{t+1} \;=\; w_t + \alpha_w\,\delta_t\,\nabla_w v(s_t, w_t)
$$

Actor (policy update):
与 QAC 一样，以 $J(\theta)$ 为目标的梯度上升
$$
\theta_{t+1} \;=\; \theta_t + \alpha_{\theta}\,\delta_t\,\nabla_{\theta}\ln\pi(a_t\mid s_t,\theta_t)
$$


#### importance sampling

之前用到 $\nabla_{\theta}J(\theta)=\mathbb{E}_{S\sim\eta,\,A\sim\pi}\!\left[*\right]$, 其中的 A 要求服从的就是 target $\pi$ ，即所用的数据得是 target $\pi$

importance sampling 可以把一个 on-policy 的算法转换为 off-policy 的算法，
实际上，importance sampling technique is not limited to AC, but also to any algorithm that aims to estimate an expectation.

$$
\begin{aligned}
\mathbb{E}_{X\sim p_0}[X]
&=\sum_{x} p_0(x)\,x \\
&=\sum_{x} p_1(x)\,\frac{p_0(x)}{p_1(x)}\,x \\
&=\mathbb{E}_{X\sim p_1}\!\left[\frac{p_0(X)}{p_1(X)}\,X\right].
\end{aligned}
$$
核心思想就是想办法把 behavior policy 采集的数据最后稳定的期望转为 target policy 的期望，从而符合 $\nabla_{\theta}J(\theta)=\mathbb{E}_{S\sim\eta,\,A\sim\pi}\!\left[*\right]$ 的要求；

令 $f(X)=\frac{p_0(X)}{p_1(X)}\,X$ ，$\mathbb{E}_{X\sim p_0}[X]$ 是我们想要的东西，$p_0$ 是 target 的分布，但是我们在 $p_1$ 下采数据，只需要把$p_1$ 下采得的数据过一下$f(\cdot)$ ，用 
$$
\bar f
=\frac{1}{n}\sum_{i=1}^{n} f(x_i)
=\frac{1}{n}\sum_{i=1}^{n}\frac{p_0(x_i)}{p_1(x_i)}\,x_i.
$$
即可近似 $\mathbb{E}_{X\sim p_0}[X]$

#### Deterministic actor-critic

概率化 $\pi(a \mid s, \theta)$ 无法应对 action 有无数种的情况，因此 continuous 的情况：
$$
a=\mu(s,\theta)\;:=\;\mu(s).
$$
也就是策略 $\mu$   直接映射出具体的 action $a$

思路跟上面一样，两部分：
1. 梯度是啥  
2. 有了梯度具体算法是啥，如何更新参数

$$
J(\theta) = \mathbb{E}\!\left[v_\mu(s)\right]
= \sum_{s\in S} d_0(s)\, v_\mu(s)
$$

$$
\nabla_\theta J(\theta)
= \sum_{s\in S} \rho_\mu(s)\, \nabla_\theta \mu(s)\,
\left.\bigl(\nabla_a q_\mu(s,a)\bigr)\right|_{a=\mu(s)}
= \mathbb{E}_{S\sim \rho_\mu}\!\left[
\nabla_\theta \mu(S)\,
\left.\nabla_a q_\mu(S,a)\right|_{a=\mu(S)}
\right]
$$
这个梯度的推导过程可以理解为链式法则：
$$
\nabla_\theta v_\mu(s) = \nabla_\theta q_\mu\!\bigl(s,\mu_\theta(s)\bigr)
$$

参数的理论梯度上升更新式子为：
$$
\theta_{t+1} = \theta_t + \alpha_\theta \,\mathbb{E}_{S\sim \rho_\mu}\!\left[
\nabla_\theta \mu(S)\,\left.(\nabla_a q_\mu(S,a))\right|_{a=\mu(S)}
\right]
$$
实操还是 MC stochastic gradient 采样：
$$
\theta_{t+1} = \theta_t + \alpha_\theta \,\nabla_\theta \mu(s_t)\,\left.(\nabla_a q_\mu(s_t,a))\right|_{a=\mu(s_t)}
$$





---


# PPO 

policy gradient 两种等价写法：
$$
\nabla_{\theta}J(\theta) =\mathbb{E}_{(S_t, A_t)\sim d_{\pi_\theta}(s)\,\pi_\theta(a\mid s)}\!\left[\nabla_{\theta}\ln \pi(A\mid S,\theta)\,q_{\pi}(S,A)\right]
$$

和
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim P_\theta(\tau)}\left[\nabla_\theta \log \pi_\theta(a_t\mid s_t),R(\tau)\right]
$$
其中 $\tau$ 表示一条轨迹，$R(\tau)$ 表示这条轨迹的 reward

这俩等价也不用啥数学推导，第一个式子里面是一个二元随机变量，A 取不同值就囊括了该状态下所有可能的 trajectory




---
