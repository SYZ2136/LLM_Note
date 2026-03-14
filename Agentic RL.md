
# RL 更新 LLM 训练框架


#### 采样数据

假设一个 batch 有如下两条 prompt
`q1：今天天气怎么样`
`q2：帮我打个招呼`
 开始采样，当前 actor model 为初始参数 $\theta_0$, 得到这一批次的采样数据 (query, response)
 
$(q_1, r_1)$：今天天气怎么样 $\to$ 真 不错
$(q_2, r_2)$：帮我打个招呼 $\to$ 你 好呀

$$
r_1 = (o_{1,1}, o_{1,2}) = (\text{真}, \text{不错})
$$

$$
r_2 = (o_{2,1}, o_{2,2}) = (\text{你}, \text{好呀})
$$

这个 batch 采样完成会保存这个 batch 采样的 $\pi_{old}$

对于第一条样本 $(q_1, r_1)$

第 1 个 token：

$$
\pi_{old}^{(1,1)}
=
\pi_{\theta_0}(o_{1,1} \mid q_1)
=
\pi_{\theta_0}(\text{真} \mid \text{今天天气怎么样})
$$

第 2 个 token：

$$
\pi_{old}^{(1,2)}
=
\pi_{\theta_0}(o_{1,2} \mid q_1, o_{1,1})
=
\pi_{\theta_0}(\text{不错} \mid \text{今天天气怎么样真})
$$

对于第二条样本 $(q_2, r_2)$

第 1 个 token：

$$
\pi_{old}^{(2,1)}
=
\pi_{\theta_0}(o_{2,1} \mid q_2)
=
\pi_{\theta_0}(\text{你} \mid \text{帮我打个招呼})
$$

第 2 个 token：

$$
\pi_{old}^{(2,2)}
=
\pi_{\theta_0}(o_{2,2} \mid q_2, o_{2,1})
=
\pi_{\theta_0}(\text{好呀} \mid \text{帮我打个招呼你})
$$

工程中更常保存的是：

$$
\log \pi_{old}^{(i,t)}
=
\log \pi_{\theta_0}(o_{i,t} \mid q_i, o_{i,<t})
$$




#### 单个 rollout 会用多次

$(q_1, r_1)$ 会 update 多次模型参数，增加数据利用率，比如一个数据用三次

对于同一条 rollout 数据，如果 `update_times = 3`，那么

$\pi_{old}$ 在这 3 次里都是同一个，始终等于最开始采样这条数据时得到的
而这 3 次对应的

$$
\pi_{new,1},\ \pi_{new,2},\ \pi_{new,3}
$$

则是在第 1、2、3 次 update 时，当前 actor 对**这条最开始采样得到的固定轨迹**重新计算出来的概率。

$$
\pi_{new,1} = \pi_{old} \qquad \pi_{new,2} \ne \pi_{old} \qquad \pi_{new,3} \ne \pi_{old}
$$



###### 第 1 次 update：
actor 还没有更新，因此当前参数仍然是：

$$
\theta_0
$$

即
$$
r = \frac{\pi_{new}}{\pi_{old}} = 1
$$

**第一次没有体现出重要性采样**，但是第一次 update 后，模型参数为：

$$
\theta_0 \rightarrow \theta_1
$$


####### 第 2 次 update

当前 actor 参数变为：

$$
\theta_1
$$

那么在这个新参数下，按着采样的那个 rollout 去走得到的 $\pi_{new}$ 就与 $\pi_{old}$ 不同了

$$
\pi_{new,2}^{(i,t)}
=
\pi_{\theta_1}(o_{i,t} \mid q_i, o_{i,<t})
$$

这个批次内

$$
\pi_{new,2}^{(1,1)}
=
\pi_{\theta_1}(\text{真} \mid \text{今天天气怎么样})
$$

$$
\pi_{new,2}^{(1,2)}
=
\pi_{\theta_1}(\text{不错} \mid \text{今天天气怎么样}, \text{真})
$$

$$
\pi_{new,2}^{(2,1)}
=
\pi_{\theta_1}(\text{你} \mid \text{帮我打个招呼})
$$

$$
\pi_{new,2}^{(2,2)}
=
\pi_{\theta_1}(\text{好呀} \mid \text{帮我打个招呼}, \text{你})
$$

第二次 update 后模型参数：
$$
\theta_1 \rightarrow \theta_2
$$


#### LLM 参数更新 

实际工程训练中是批次为单位：
一批 rollout →  前向传播计算 batch loss → batch loss 反向传播更新参数

但是逻辑理解上，一条 rollout 为单位去理解
**一条 rollout 中间各个时间步各种复杂的公式，不论是 TD-error， GAE 等等，本质都是为了获得这一条 rollout 对应的 loss 的中间量**

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
其中 $\tau$ 表示一条轨迹，$R(\tau)$ 表示这条轨迹的 reward；这俩等价也不用啥数学推导，第一个式子里面是一个二元随机变量，A 取不同值就囊括了该状态下所有可能的 trajectory


#### clip

$$
L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[
\min\left(
r_t(\theta)\hat{A}_t,\;
\operatorname{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\hat{A}_t
\right)
\right]
$$

$$
r_t(\theta)=\frac{\pi_{\text{new}}(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$


$$
\operatorname{clip}(x, a, b)=
\begin{cases}
a, & x<a \\\\
x, & a\le x\le b \\\\
b, & x>b
\end{cases}
$$

- $\hat{A}_t > 0$，应该奖励该动作
- $\hat{A}_t < 0$,  应该惩罚该动作

$\min\left(r_t(\theta)\hat{A}_t,\;\operatorname{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\hat{A}_t\right)$ , 外面整体取一个 min，$\hat{A}_t > 0$ 时，限制奖励上限 $1 +\epsilon$ 并且 $\hat{A}_t < 0$ 时，限制惩罚上限 $1 - \epsilon$；具体证明画函数图就行，$r_t(\theta)$ 看成 $x$，其实就是 $y = x$ 分段函数取 min (A > 0 )或者 max (A < 0) 的问题


#### 工程实现

$$
\mathcal{L}_{\text{policy}}(\theta)
=
-
\mathbb{E}_{q,\;o\sim \pi_{\theta_{\text{old}}}}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\min\left(
r_t(\theta)A_t,\;
\operatorname{clip}\big(r_t(\theta),1-\epsilon,1+\epsilon\big)A_t
\right)
\right]
$$



其中 Generalized advantage estimation (GAE) 计算过程为：


