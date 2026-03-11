
# PPO


## clip

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