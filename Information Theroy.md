
许多学科用 $\log(x)$ 代替 $\log_{b}(x)$，$b$ 的值根据上下文确定
信息论中涉及熵，底数多取 2，单位为bit；
`Pytorch` 框架中各种熵的实现底数为 e，不同底数本质只是乘一个常数系数

---


## Self-information
在随机变量 $X$ 的某个具体观测值下获得了多少信息$$I(X = x_k) = -\log_a p(x_k)$$以2为底的时候单位为 bit

---


## Entropy
$$H(P) = - \mathbb{E}_{X \sim P(x)}\big[\log_{2} P(x)\big]$$
熵越大，不确定性越高，表示的是**在 $X \sim P(x)$ 这个概率分布下，每观察一次 $X$ ，平均能获得多少信息**
观测前的不确定度越高 $\Leftrightarrow$ 观测后提供的信息量越多

---


## Shannon 最优编码
若真实分布为 P , 用它设计的最优前缀码，熵 $H(P)$ 是平均码长的下界

---


## Cross Entropy
随机变量 X 的真实分布为 P , 模型预测为 Q，
每次的预测值给出的信息量为 $-\log_{2}Q(x)$，
$-\log_{2}Q(x)$ 本身可以看成随机变量 X 的函数，
交叉熵：**真实分布为 P 时，按照 Q 预测，每次预测平均提供的信息量**
$$H(P, Q) = \mathbb{E}_{x \sim P(X)}\big[-\log_{2} Q(x)\big]$$**供的信息量越少，说明用 Q 预测 P 消除的不确定度越小**

---


## KL divergence
$$D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}\big[-\log Q(x)-(-\log P(x)) \,\, ]$$

$$
D_{\mathrm{KL}}(P\|Q)=\sum_{x} P(x)\,\log\frac{P(x)}{Q(x)}.
$$
- KL 散度非负 
- 随机变量 X 的真实分布为 P , 用 Q 去编码，最优编码下界上移了多少
- Q 的每次预测平均提供的信息量与 P 的每次真实观测平均提供的信息量差值

KL 散度是不对称度量，用 q 去拟合 p，
可以选择用 $KL(p||q)$ , 比如多分类任务的交叉熵
也可以选择用 $KL(q||p)$，比如 VAE 中高斯分布拟合隐变量后验
选 $KL(p||q)$ 还是 $KL(q||p)$ 区别在于，
1. 期望的底数你选谁，你的问题实际进行算法优化是可操作的
2. 两者理论上都是 q 与 p 越接近，损失越小，但是二者拟合的偏好不一样；
    $KL(p||q)$ ，p 作为期望底数，那么 p 大的时候，q 一定尽量大，但是 p 小的时候，q 不一定要很小
    $KL(q||p)$ ， q 作为期望底数，p 小的时候，q一定要小，否则惩罚非常大；



---


## Mutual information
**知道 Y 之后，X 的不确定度减少多少**
$$I(X;Y)=\mathbb{E}_{(x,y)\sim P_{X,Y}}\big[ -\log\big(P_X(x)P_Y(y)\big) - \big(-\log P_{X,Y}(x,y)\big)\big]$$
$$\begin{align*}I(X;Y) &= H(X) - H(X \mid Y) \\
       &= H(Y) - H(Y \mid X) \\
       &= H(X) + H(Y) - H(X,Y)\end{align*}$$
*Proof* 

$$
\begin{aligned}
H(X,Y)
&= - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log p(x,y) \\
&= - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log\bigl(p(x)p(y\mid x)\bigr) \\
&= - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log p(x)
   - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log p(y\mid x) \\
&= - \sum_{x \in \mathcal{X}} p(x)\log p(x)
   - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log p(y\mid x) \\
&= H(X) + H(Y\mid X).
\end{aligned}
$$


---


