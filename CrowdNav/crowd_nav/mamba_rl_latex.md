# MambaRL Policy LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - MambaRL Policy}
\begin{itemize}
  \item $\mathbf{s}_r \in \mathbb{R}^{9}$: Robot state [px, py, gx, gy, vx, vy, radius, v\_pref, theta]
  \item $\mathbf{s}_h^{(i)} \in \mathbb{R}^{5}$: Human $i$ state [px, py, vx, vy, radius]
  \item $\mathbf{s} \in \mathbb{R}^{34}$: Joint state $[\mathbf{s}_r; \mathbf{s}_h^{(1)}; ...; \mathbf{s}_h^{(5)}]$
  \item $\mathbf{h} \in \mathbb{R}^{H}$: Hidden dimension ($H = 64$ from config)
  \item $K$: Top-K nearest humans to consider ($K = 3$ from policy.config)
  \item $N = 5$: Maximum number of humans
  \item $\gamma = 0.95$: Discount factor (from train.config)
  \item $\sigma_{\text{scale}} = 0.65$: Action scale factor (from policy.config)
  \item $\epsilon$: Exploration noise parameter
  \item $\pi_\theta(\mathbf{a}|\mathbf{s})$: Actor policy network
  \item $Q_\phi(\mathbf{s}, \mathbf{a})$: Critic Q-value networks (twin Q1, Q2)
  \item $\mu_\theta(\mathbf{s}), \sigma_\theta(\mathbf{s})$: Mean and log-std from actor
\end{itemize}

\section*{State Preprocessing}
\begin{algorithm}[H]
\caption{State Splitting and Relative Coordinate Transformation}
\KwIn{Joint state $\mathbf{s} \in \mathbb{R}^{34}$}
\KwOut{Robot features $\mathbf{r} \in \mathbb{R}^{1 \times 9}$, Human features $\mathbf{H} \in \mathbb{R}^{N \times 5}$, Mask $\mathbf{M} \in \{0,1\}^N$}

$\mathbf{r} \leftarrow \mathbf{s}[:9]$ \tcp{Extract robot state}
$\mathbf{H} \leftarrow \text{reshape}(\mathbf{s}[9:], (N, 5))$ \tcp{Extract human states}
$\mathbf{p}_r \leftarrow \mathbf{r}[:2]$ \tcp{Robot position (px, py)}

\For{$i = 1$ to $N$}{
    \If{$\mathbf{H}[i]$ is valid}{
        $\mathbf{M}[i] \leftarrow 1$
        $\mathbf{H}[i][:2] \leftarrow \mathbf{H}[i][:2] - \mathbf{p}_r$ \tcp{Relative coordinates}
    }
    \Else{
        $\mathbf{M}[i] \leftarrow 0$
    }
}

\tcp{Distance-based sorting}
$\mathbf{d} \leftarrow \|\mathbf{H}[:, :2]\|_2$ \tcp{Distances to robot}
$\text{idx} \leftarrow \text{argsort}(\mathbf{d})$
$\mathbf{H} \leftarrow \mathbf{H}[\text{idx}]$, $\mathbf{M} \leftarrow \mathbf{M}[\text{idx}]$

\tcp{Top-K selection}
\If{$K < N$}{
    $\mathbf{H}[K:] \leftarrow \mathbf{0}$
    $\mathbf{M}[K:] \leftarrow 0$
}
\end{algorithm}

\section*{Mamba State Encoder}
\begin{align}
\mathbf{t}_r &= \text{RobotEncoder}(\mathbf{r}) = \text{SiLU}(\text{LN}(\mathbf{W}_r \mathbf{r} + \mathbf{b}_r)) \\
\mathbf{t}_h^{(i)} &= \text{HumanEncoder}(\mathbf{H}[i]) = \text{SiLU}(\text{LN}(\mathbf{W}_h \mathbf{H}[i] + \mathbf{b}_h)) \\
\mathbf{T} &= [\mathbf{t}_r; \mathbf{t}_h^{(1)}; ...; \mathbf{t}_h^{(N)}] \in \mathbb{R}^{(1+N) \times H}
\end{align}

\begin{algorithm}[H]
\caption{Mamba Block Processing}
\KwIn{Token sequence $\mathbf{T} \in \mathbb{R}^{(1+N) \times H}$, Number of blocks $L = 4$}
\KwOut{Processed sequence $\mathbf{T}^{(L)}$}

\For{$\ell = 1$ to $L$}{
    $\hat{\mathbf{T}}^{(\ell)} \leftarrow \text{LayerNorm}(\mathbf{T}^{(\ell-1)})$
    $\mathbf{T}^{(\ell)} \leftarrow \mathbf{T}^{(\ell-1)} + \text{Mamba}(\hat{\mathbf{T}}^{(\ell)})$
    $\mathbf{T}^{(\ell)} \leftarrow \text{Dropout}(\mathbf{T}^{(\ell)})$
}
\end{algorithm}

\section*{Attention Pooling}
\begin{align}
\mathbf{Q} &= \mathbf{t}_r \mathbf{W}_Q \in \mathbb{R}^{1 \times H} \\
\mathbf{K} &= \mathbf{T}_h \mathbf{W}_K \in \mathbb{R}^{N \times H} \\
\mathbf{V} &= \mathbf{T}_h \mathbf{W}_V \in \mathbb{R}^{N \times H} \\
\mathbf{A} &= \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{H}} \odot \mathbf{M}\right) \\
\mathbf{c} &= \mathbf{A}\mathbf{V} \in \mathbb{R}^{H} \\
\mathbf{f} &= [\mathbf{t}_r; \mathbf{c}] \in \mathbb{R}^{2H}
\end{align}

\section*{Actor Network (TanhGaussian)}
\begin{align}
\mathbf{z} &= \text{SiLU}(\text{LN}(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1)) \\
\mathbf{h}_a &= \text{SiLU}(\mathbf{W}_2 \mathbf{z} + \mathbf{b}_2) \\
\boldsymbol{\mu} &= \mathbf{W}_\mu \mathbf{h}_a + \mathbf{b}_\mu \\
\log \boldsymbol{\sigma} &= \text{clamp}(\mathbf{W}_\sigma \mathbf{h}_a + \mathbf{b}_\sigma, -5, 2) \\
\boldsymbol{\sigma} &= \exp(\log \boldsymbol{\sigma}) \\
\mathbf{u} &\sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}) \\
\mathbf{a} &= \tanh(\mathbf{u}) \\
\log \pi(\mathbf{a}|\mathbf{s}) &= \sum_{j=1}^{2} \left[\log \mathcal{N}(u_j|\mu_j, \sigma_j) - \log(1 - a_j^2 + \epsilon)\right]
\end{align}

\section*{Critic Networks (Twin Q)}
\begin{align}
\mathbf{x}_{qa} &= [\mathbf{f}; \mathbf{a}] \in \mathbb{R}^{2H+2} \\
\mathbf{h}_{q1} &= \text{SiLU}(\text{LN}(\mathbf{W}_{q1}^{(1)} \mathbf{x}_{qa} + \mathbf{b}_{q1}^{(1)})) \\
\mathbf{h}_{q1} &= \text{SiLU}(\mathbf{W}_{q1}^{(2)} \mathbf{h}_{q1} + \mathbf{b}_{q1}^{(2)}) \\
Q_1(\mathbf{s}, \mathbf{a}) &= \mathbf{W}_{q1}^{(3)} \mathbf{h}_{q1} + b_{q1}^{(3)} \\
Q_2(\mathbf{s}, \mathbf{a}) &= \text{similar to } Q_1 \text{ with different parameters}
\end{align}

\section*{Action Scaling and Output}
\begin{align}
\mathbf{a}_{\text{raw}} &\in [-1, 1]^2 \text{ (from tanh)} \\
v_{\text{pref}} &= 1.0 \text{ (from env.config robot section)} \\
\sigma_{\text{scale}} &= 0.65 \text{ (from policy.config)} \\
\mathbf{a}_{\text{phys}} &= \mathbf{a}_{\text{raw}} \cdot v_{\text{pref}} \cdot \sigma_{\text{scale}} \\
\text{ActionXY}(v_x, v_y) &= \text{ActionXY}(a_{\text{phys}}[0], a_{\text{phys}}[1])
\end{align}

\section*{Observation Normalization}
\begin{align}
\boldsymbol{\mu}_n &= \frac{1}{n}\sum_{i=1}^n \mathbf{s}_i \text{ (running mean)} \\
\boldsymbol{\sigma}_n^2 &= \frac{1}{n}\sum_{i=1}^n (\mathbf{s}_i - \boldsymbol{\mu}_n)^2 \text{ (running variance)} \\
\hat{\mathbf{s}} &= \frac{\mathbf{s} - \boldsymbol{\mu}_n}{\sqrt{\boldsymbol{\sigma}_n^2 + \epsilon}}
\end{align}

\section*{Training vs Inference Modes}
\begin{algorithm}[H]
\caption{Action Selection Strategy}
\KwIn{Observation $\mathbf{s}$, Phase $\in \{\text{train}, \text{test}\}$, Stochastic flag}
\KwOut{Action $\mathbf{a}$}

$\hat{\mathbf{s}} \leftarrow \text{normalize}(\mathbf{s})$
$\mathbf{f} \leftarrow \text{encode}(\hat{\mathbf{s}})$

\If{phase == 'train' AND env\_stochastic}{
    $\mathbf{a}, \log\pi, \mathbf{a}_{\text{det}} \leftarrow \text{actor.sample}(\mathbf{f})$
}
\Else{
    $\boldsymbol{\mu}, \log\boldsymbol{\sigma} \leftarrow \text{actor}(\mathbf{f})$
    $\mathbf{a} \leftarrow \tanh(\boldsymbol{\mu})$ \tcp{Deterministic}
}

\If{phase == 'train' AND $\epsilon > 0$}{
    $\mathbf{a} \leftarrow \text{clip}(\mathbf{a} + \mathcal{N}(0, \epsilon), -1, 1)$ \tcp{Exploration noise}
}

$\mathbf{a}_{\text{scaled}} \leftarrow \mathbf{a} \cdot v_{\text{pref}} \cdot \sigma_{\text{scale}}$
\Return{ActionXY($a_{\text{scaled}}[0]$, $a_{\text{scaled}}[1]$)}
\end{algorithm}

\section*{Configuration Parameters}
From \texttt{policy.config}:
\begin{itemize}
  \item \texttt{hidden\_dim = 64}: Mamba hidden dimension $H$
  \item \texttt{n\_blocks = 4}: Number of Mamba blocks $L$
  \item \texttt{topk = 3}: Keep only $K$ nearest humans
  \item \texttt{action\_scale = 0.65}: Action scaling factor $\sigma_{\text{scale}}$
  \item \texttt{conv\_dim = 4}: Mamba convolution dimension
  \item \texttt{expand = 2}: Mamba expansion factor
  \item \texttt{dropout = 0.0}: Dropout probability
\end{itemize}

From \texttt{train.config}:
\begin{itemize}
  \item \texttt{gamma = 0.95}: Discount factor $\gamma$
\end{itemize}

From \texttt{env.config}:
\begin{itemize}
  \item \texttt{v\_pref = 1.0}: Robot preferred velocity $v_{\text{pref}}$
  \item \texttt{kinematics = holonomic}: Robot kinematics model
\end{itemize}

\end{document}
```