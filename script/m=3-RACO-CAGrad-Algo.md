\begin{algorithm}[h]
\caption{RACO with CAGrad-Clip}
\label{alg:raco-cagrad-k2}
\begin{algorithmic}[1]
\REQUIRE $w\in\Delta_m$, $c\in[0,1)$, stepsize $\eta>0$.
\FOR{$t=0,1,\ldots,T$}
  \STATE Sample minibatch $\mathcal{B}_t$ of preference pairs $(x,y^a,y^b)$. 
  \STATE For each $i\in[m]$, compute loss $\mathcal{L}_i(\theta_t)$ on $\mathcal{B}_t$ and gradient $g_i^{(t)}\gets\nabla_\theta \mathcal{L}_i(\theta)\big|_{\theta=\theta_t}$.
  \STATE Compute weighted gradient $g_0^{(t)}\gets\sum_{i=1}^m w_i\,g_i^{(t)}$.
  \STATE Solve {\small $p^{(t)}\in\arg\min\limits_{p\in\Delta_m}\left\{G_p^{(t)\top} g_0^{(t)}+c\|g_0^{(t)}\|\|G_p^{(t)}\|\right\}$}, where $G_p^{(t)}:=\sum_{i=1}^m p_i g_i^{(t)}$. 
  \STATE Clip coefficients elementwise: $\tilde p^{(t)}\gets \min\{p^{(t)},w\}$. 
  \STATE Form clipped mixture $\widetilde G_p^{(t)}\gets\sum_{i=1}^m \tilde p_i^{(t)} g_i^{(t)}$.
  \STATE 
  Set $G_0^{(t)}\gets \left\{ \begin{array}{@{}ll@{}} 
    \textstyle g_0^{(t)}+c\|g_0^{(t)}\|\tfrac{\widetilde G_p^{(t)}}{\|\widetilde G_p^{(t)}\|} & \text{if $\|\widetilde G_p^{(t)}\|> 0$}, \\
    \textstyle g_0^{(t)} & \text{otherwise} 
    \end{array} \right.$
  \STATE Update $\theta_{t+1}\gets\theta_t-\eta\,G_0^{(t)}$.
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Efficient Solution towards Line~5 of Algorithm~\ref{alg:raco-cagrad-k2}}\label{app:close}

\paragraph{$m=2$ case.} The subproblem in line~5 of Algorithm~\ref{alg:raco-cagrad-k2} can be solved efficiently even in the high-dimensional LLM policy space. We give a closed-form derivation for the case of two objectives (i.e., $m=2$). Fix an iteration $t$. To simplify notation, we drop the superscript $(t)$ in this subsection (e.g., $g_i:=g_i^{(t)}$ and $g_0:=g_0^{(t)}$). Let $p=(\lambda,1-\lambda)$ with $\lambda\in[0,1]$, and let $b=(b_1,b_2)$ where $b_i=\langle g_i,g_0\rangle$ with $\delta:=b_1-b_2$. Define
\[
H=\begin{bmatrix}
    H_{11} & H_{12} \\
    H_{12} & H_{22}
\end{bmatrix} \text{ where $H_{ij}=\langle g_i,g_j\rangle$,}
\quad s=c\|g_0\|.
\]
Compute the quadratic $Q(\lambda):=q_2\lambda^2+q_1\lambda+q_0$, where
\[
q_2:= H_{11}+H_{22}-2H_{12}, \quad q_1:=2(H_{12}-H_{22}), \quad q_0:=H_{22}.
\]
The resulting one-dimensional objective is
\[
h(\lambda)=b_2+\delta\lambda+s\sqrt{Q(\lambda)}.
\]
Setting $h'(\lambda)=0$, it suffices to solve
\[
(\delta^2q_2-s^2q_2^2)\lambda^2+(\delta^2q_1-s^2q_1q_2)\lambda+\delta^2q_0-\frac{s^2q_1^2}{4}=0.
\]
This quadratic has a closed-form solution and at most two real roots. We retain the roots in $[0,1]$ and evaluate $h(\lambda)$ at each candidate. We also evaluate the endpoints $h(0)$ and $h(1)$, and select the minimizer among all candidates.

\paragraph{$m=3$ case.} Let $\Phi(p):=b^\top p+s\sqrt{p^\top Hp}$. A  subgradient of $\Phi$ is
\[
d(p) = \begin{cases}
    b + s \dfrac{Hp}{\sqrt{p^\top H p}}, & p^\top H p > 0, \\
    b, & p^\top H p = 0.
\end{cases}
\]
The second line is valid because when $G_p = 0$, the subdifferential of $\|G_p\|$ contains $0$. Use negative entropy on the simplex. The mirror step is 
\[
p^{(t+1)} = \arg\min_{p \in \Delta_m} \left\{ \eta \langle d^{(t)}, p \rangle + D_{\mathrm{KL}}(p \| p^{(t)}) \right\},
\]
whose closed form is
\[
p_i^{(t+1)} = \frac{p_i^{(t)} \exp(-\eta d_i^{(t)})}{\sum_{j=1}^{m} p_j^{(t)} \exp(-\eta d_j^{(t)})}.
\]

\textcolor{blue}{Another method.} Exact active-set solver: for $S\subset \{1,\ldots,m\}$, define 
\[
    \Delta_S := \{ p \in \Delta_m : p_i = 0 \text{ for } i \notin S \},
\]
Let $H_{SS}$ and $b_S$ be the principal submatrix and subvector respectively, and let $\mathbf{1}_S$ be the all-one vector of length $|S|$. 


\textsc{Line5}$(g_1, \ldots, g_m, w, c)$
\begin{enumerate}
    \item Compute $g_0 = \sum_i w_i g_i$.
    \item If $\|g_0\| = 0$, return any $p \in \Delta_m$, e.g.\ $p = w$.
    \item Build
    \[
    H_{ij} = \langle g_i, g_j \rangle, \quad b = Hw, \quad s = c\|g_0\|.
    \]
    \item Return \textsc{Face}$([m])$.
\end{enumerate}

\medskip
\noindent
\textsc{Face}$(S)$ \\[0.5em]
\textbf{Input:} support $S$ \\
\textbf{Output:} exact minimizer of $\Phi$ over $\Delta_S$

\begin{enumerate}[label=\Alph*.]
    \item If \textsc{Face}$(S)$ is already memoized, return it.
    \item Initialize candidate list $\mathcal{C} = \emptyset$.
    \item Interior positive-norm candidates:
    \begin{enumerate}[label=\arabic*.]
        \item Form $\psi_S(\nu) := (\nu \mathbf{1}_S - b_S)^\top H_{SS}^\dagger (\nu \mathbf{1}_S - b_S) - s^2$.
        \item Solve $\psi_S(\nu) = 0$. This gives at most two real roots.
        \item For each real root $\nu$: \\
        solve
        \[
        \begin{aligned}
        H_{SS} p_S &= \lambda (\nu \mathbf{1}_S - b_S), \\
        \mathbf{1}_S^\top p_S &= 1, \\
        p_S &> 0, \\
        \lambda &> 0.
        \end{aligned}
        \]
        If feasible: \\
        \hspace*{1em} embed $p_S$ into a full $p \in \Delta_m$ by setting $p_i = 0$ for $i \notin S$, \\
        \hspace*{1em} add $(p, \Phi(p) = \nu)$ to $\mathcal{C}$.
    \end{enumerate}
    \item Interior zero-norm candidate: \\
    solve
    \[
    \begin{aligned}
    H_{SS} p_S &= 0, \\
    \mathbf{1}_S^\top p_S &= 1, \\
    p_S &> 0.
    \end{aligned}
    \]
    If feasible: \\
    \hspace*{1em} embed into $p \in \Delta_m$ and add $(p, \Phi(p) = 0)$ to $\mathcal{C}$.
    \item Boundary recursion: \\
    If $|S| > 1$, then for each $j \in S$: \\
    \hspace*{1em} add \textsc{Face}$(S \setminus \{j\})$ to $\mathcal{C}$.
    \item Return the candidate in $\mathcal{C}$ with smallest $\Phi$-value, and memoize it.
\end{enumerate}


For example, when $m=3$, the candidate set consists of: the 3 vertices; the 3 edge minimizers; and the interior candidate(s), if any. Because $\Phi$ is convex on $\Delta_3$, every global minimizer lies either at a vertex, on an edge, or in the relative interior. So this list is exhaustive.

First, for 3 vertices, evaluate 
\[
\Phi(e_i) = b_i + s\sqrt{H_{ii}} = b_i + s\|g_i\|, \qquad i = 1, 2, 3,
\]
where $e_1 = (1,0,0)$, $e_2 = (0,1,0)$, $e_3 = (0,0,1)$.

Then, for 3 edges, i.e., each unordered pair $\{i,j\} \subset \{1,2,3\}$, let $k$ be the remaining index and restrict to the edge
\[
p_k = 0, \qquad p_i = \lambda, \qquad p_j = 1 - \lambda, \qquad \lambda \in [0,1].
\]
Then
\[
\Phi_{ij}(\lambda) = b_j + \delta_{ij}\lambda + s\sqrt{Q_{ij}(\lambda)}, \qquad \delta_{ij} := b_i - b_j,
\]
with
\[
Q_{ij}(\lambda) = q_2^{ij} \lambda^2 + q_1^{ij} \lambda + q_0^{ij},
\]
where
\[
q_2^{ij} := H_{ii} + H_{jj} - 2H_{ij}, \qquad q_1^{ij} := 2(H_{ij} - H_{jj}), \qquad q_0^{ij} := H_{jj}.
\]
This is exactly the paper's $m=2$ formula, applied to the restricted pair $(i,j)$. The stationary points are the real roots in $[0,1]$ of
\[
\big((\delta_{ij})^2 q_2^{ij} - s^2 (q_2^{ij})^2\big)\lambda^2 + \big((\delta_{ij})^2 q_1^{ij} - s^2 q_1^{ij} q_2^{ij}\big)\lambda + (\delta_{ij})^2 q_0^{ij} - \frac{s^2 (q_1^{ij})^2}{4} = 0.
\]
For each edge $\{i,j\}$, keep all real roots in $[0,1]$, evaluate $\Phi_{ij}$ there, also evaluate the endpoints $\lambda = 0, 1$, and keep the edge minimizer. Do this for $\{1,2\}$, $\{1,3\}$, and $\{2,3\}$.

Finally, for interior candidate, i.e., $p_i > 0$ for all $i$, assume first that $p \in \mathrm{ri}(\Delta_3)$ and $p^\top H p > 0$. Then the equality-constrained KKT condition is
\[
b + s \frac{Hp}{\sqrt{p^\top H p}} = \nu \mathbf{1}, \qquad \mathbf{1}^\top p = 1,
\]
for some scalar $\nu$, where $\mathbf{1} = (1,1,1)^\top$. Multiplying by $p^\top$ gives $\Phi(p) = \nu$, so any interior candidate has objective value $\nu$. 

Now we need to consider two cases, if $H$ is invertible, define
\[
A := \mathbf{1}^\top H^{-1} \mathbf{1}, \qquad B := \mathbf{1}^\top H^{-1} b, \qquad C := b^\top H^{-1} b.
\]
Then every interior KKT point must satisfy
\[
p(\nu) = \frac{H^{-1}(\nu \mathbf{1} - b)}{\mathbf{1}^\top H^{-1}(\nu \mathbf{1} - b)} = \frac{H^{-1}(\nu \mathbf{1} - b)}{A\nu - B},
\]
where $\nu$ solves
\[
(\nu \mathbf{1} - b)^\top H^{-1} (\nu \mathbf{1} - b) = s^2,
\]
that is,
\[
A\nu^2 - 2B\nu + C - s^2 = 0.
\]
So: solve the scalar quadratic above; for each real root $\nu$, compute $p(\nu)$; keep it only if
\[
p_1(\nu) > 0, \qquad p_2(\nu) > 0, \qquad p_3(\nu) > 0.
\]
Each such $p(\nu)$ is an interior candidate, with objective value $\Phi(p(\nu)) = \nu$.

If $H$ is singular, then the correct complete test is: a positive-norm interior candidate must satisfy 
\[
Hp = \alpha(\nu \mathbf{1} - b), \qquad \mathbf{1}^\top p = 1, \qquad p > 0, \qquad \alpha > 0,
\]
for some scalar $\nu$, together with
\[
(\nu \mathbf{1} - b)^\top H^\dagger (\nu \mathbf{1} - b) = s^2
\]
and the consistency condition
\[
(I - HH^\dagger)(\nu \mathbf{1} - b) = 0.
\]
Since everything is only $3 \times 3$, the implementation is tiny: solve the scalar equation
\[
(\nu \mathbf{1} - b)^\top H^\dagger (\nu \mathbf{1} - b) = s^2;
\]
for each real root $\nu$ satisfying the consistency condition, solve
\[
\begin{bmatrix}
H & -(\nu \mathbf{1} - b) \\
\mathbf{1}^\top & 0
\end{bmatrix}
\begin{bmatrix}
p \\ \alpha
\end{bmatrix}
=
\begin{bmatrix}
0 \\ 1
\end{bmatrix};
\]
keep the solution if $p_1, p_2, p_3 > 0$ and $\alpha > 0$. Again, any such interior candidate has value $\Phi(p) = \nu$.

There is one nondifferentiable case that must also be checked explicitly:
\[
p \in \mathrm{ri}(\Delta_3), \qquad p^\top H p = 0.
\]
Since $H \succeq 0$, this is equivalent to
\[
Hp = 0, \qquad \mathbf{1}^\top p = 1, \qquad p > 0.
\]
If this system is feasible, then $\|G_p\| = 0$, hence
\[
\Phi(p) = b^\top p.
\]
But $G_p = 0$ implies $b^\top p = \langle G_p, g_0 \rangle = 0$, so
\[
\Phi(p) = 0.
\]
Therefore any such point is also a valid candidate.


In conclusion, the exact $m=3$ solver is:
\[
\min\;\Phi(p) = b^\top p + s\sqrt{p^\top H p} \quad \text{over } \Delta_3
\]
by checking the following candidates in candidate set $\mathcal{C}$: 
\begin{enumerate}
    \item $e_1, e_2, e_3$;
    \item the exact minimizer on each edge $\{1,2\}, \{1,3\}, \{2,3\}$, obtained from the Appendix B.1 quadratic;
    \item every feasible interior candidate from the KKT system above;
    \item every feasible zero-norm interior candidate $Hp = 0$, $\mathbf{1}^\top p = 1$, $p > 0$.
\end{enumerate}
Then return
\[
p^\star \in \arg\min_{p \in \mathcal{C}} \Phi(p).
\]
To see why this is exact, let $p^\star$ be any global minimizer on $\Delta_3$.
\begin{itemize}
    \item If one coordinate of $p^\star$ is zero, then $p^\star$ lies on one of the 3 edges, and that edge minimizer is checked.
    \item If two coordinates are zero, then $p^\star$ is a vertex, and all 3 vertices are checked.
    \item If all coordinates are positive and $(p^\star)^\top H p^\star > 0$, then $p^\star$ satisfies the interior KKT system above.
    \item If all coordinates are positive and $(p^\star)^\top H p^\star = 0$, then $Hp^\star = 0$, so it is covered by the zero-norm interior test.
\end{itemize}
So every global minimizer is in the candidate list, hence the returned point is exact. 
