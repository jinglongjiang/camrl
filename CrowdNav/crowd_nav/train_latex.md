# Training Pipeline LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Training Pipeline}
\begin{itemize}
  \item $E_{\text{train}}$: Total training episodes (from train.config)
  \item $E_{\text{IL}} = 150$: Imitation learning episodes (from train.config)
  \item $N_{\text{BC}} = 3000$: BC pretraining iterations (from train.config)
  \item $\mathcal{D}_{\text{RL}}$: RL replay buffer (capacity = 100k)
  \item $\mathcal{D}_E$: Expert demonstration buffer
  \item $\pi_\theta$: Current policy (student)
  \item $\pi_E$: Expert policy (ORCA)
  \item $\epsilon(t)$: Exploration schedule
  \item $p_{\text{demo}}(t)$: Demo sampling ratio schedule
  \item $\lambda_{\text{BC}}(t)$: BC regularization weight schedule
  \item $\alpha(t)$: Learning rate schedule
  \item $H_{\text{target}}(t)$: Target entropy schedule
  \item $\gamma = 0.95$: Discount factor (from train.config)
  \item $\tau = 0.0075$: Polyak averaging coefficient (from train.config)
\end{itemize}

\section*{Training Hyperparameters}
From \texttt{train.config}:
\begin{align}
\text{Batch size: } &B = 1536 \\
\text{Learning rates: } &\alpha_{\text{actor}} = 2 \times 10^{-4}, \alpha_{\text{critic}} = 2.5 \times 10^{-4}, \alpha_{\text{alpha}} = 2.4 \times 10^{-4} \\
\text{Gradient clipping: } &\|\nabla\|_2 \leq 1.0 \\
\text{Target entropy: } &H_{\text{target}} = -2.4 \\
\text{BC regularization: } &\beta_{\text{AWBC}} = 2.5 \\
\text{Exploration: } &\epsilon_{\text{start}} = 0.08, \epsilon_{\text{end}} = 0.03, \epsilon_{\text{decay}} = 1500
\end{align}

\section*{Action Normalization}
\begin{align}
\mathbf{a}_{\text{phys}} &\in \mathbb{R}^2 \text{ (expert action)} \\
v_{\text{pref}} &= s[7] \text{ (robot preferred velocity from state)} \\
\sigma_{\text{scale}} &= 0.65 \text{ (from policy.config)} \\
\mathbf{a}_{\text{norm}} &= \text{clip}\left(\frac{\mathbf{a}_{\text{phys}}}{v_{\text{pref}} \cdot \sigma_{\text{scale}}}, -1, 1\right)
\end{align}

\section*{Behavioral Cloning Pretraining}
\begin{algorithm}[H]
\caption{BC Actor Pretraining}
\KwIn{Expert buffer $\mathcal{D}_E$, Policy $\pi_\theta$, Iterations $N_{\text{BC}}$}
\KwOut{Pretrained actor parameters $\theta$}

\For{$i = 1$ to $N_{\text{BC}}$}{
    Sample batch $\{(\mathbf{s}_j, \mathbf{a}_j^*)\}_{j=1}^B$ from $\mathcal{D}_E$
    
    \tcp{Normalize expert actions}
    $\mathbf{a}_j^{\text{norm}} \leftarrow \text{phys\_to\_norm\_actions}(\mathbf{a}_j^*, \mathbf{s}_j, \sigma_{\text{scale}})$
    
    \tcp{Forward pass}
    $\hat{\mathbf{s}}_j \leftarrow \text{normalize\_obs}(\mathbf{s}_j)$
    $\mathbf{h}_j \leftarrow \text{encode}(\hat{\mathbf{s}}_j)$
    $\boldsymbol{\mu}_j, \boldsymbol{\sigma}_j \leftarrow \text{actor}(\mathbf{h}_j)$
    $\mathbf{a}_j^{\mu} \leftarrow \tanh(\boldsymbol{\mu}_j)$
    
    \tcp{BC loss}
    $\mathcal{L}_{\text{BC}} \leftarrow \frac{1}{B} \sum_{j=1}^B \|\mathbf{a}_j^{\mu} - \mathbf{a}_j^{\text{norm}}\|_2^2$
    
    \tcp{Optimization}
    $\theta \leftarrow \theta - \alpha_{\text{BC}} \nabla_\theta \mathcal{L}_{\text{BC}}$
    
    \tcp{Gradient clipping}
    $\|\nabla_\theta\|_2 \leftarrow \min(\|\nabla_\theta\|_2, 1.0)$
    
    \tcp{Update observation statistics}
    Update running mean/variance with $\{\mathbf{s}_j\}$
}
\end{algorithm}

\section*{Training Schedules}
\subsection*{Exploration Schedule}
\begin{align}
t &= \frac{\text{episode}}{\max(1, \epsilon_{\text{decay}})} \\
\epsilon(t) &= \begin{cases}
\epsilon_{\text{start}} + (\epsilon_{\text{end}} - \epsilon_{\text{start}}) \cdot t & \text{if } t \leq 1 \\
\epsilon_{\text{end}} & \text{if } t > 1
\end{cases}
\end{align}

\subsection*{Target Entropy Schedule}
\begin{align}
t &= \frac{\text{episode}}{\max(1, 0.8 \cdot E_{\text{train}})} \\
H_{\text{target}}(t) &= -2.2 + (-2.8 - (-2.2)) \cdot \min(1.0, t) \\
&= -2.2 - 0.6 \cdot \min(1.0, t)
\end{align}

\subsection*{Learning Rate Schedule}
\begin{align}
p &= \frac{\text{episode}}{E_{\text{train}}} \\
\text{mult}(p) &= \begin{cases}
0.5 + 0.5 \cdot \frac{p}{0.1} & \text{if } p \leq 0.1 \text{ (warmup)} \\
1.0 & \text{if } 0.1 < p \leq 0.6 \text{ (plateau)} \\
1.0 - 0.6 \cdot \frac{p - 0.6}{0.4} & \text{if } p > 0.6 \text{ (decay)}
\end{cases} \\
\alpha_{\text{actor}}(p) &= \alpha_{\text{actor}} \cdot \text{mult}(p) \\
\alpha_{\text{critic}}(p) &= \alpha_{\text{critic}} \cdot \text{mult}(p) \\
\alpha_{\alpha}(p) &= \alpha_{\alpha} \cdot \text{mult}(p)
\end{align}

\subsection*{BC Regularization Schedule}
\begin{align}
t &= \frac{\text{episode}}{E_{\text{train}} - 1} \\
p_{\text{demo}}(t) &= 0.50 + (0.35 - 0.50) \cdot t = 0.50 - 0.15t \\
\lambda_{\text{BC}}(t) &= \begin{cases}
0.95 + (0.70 - 0.95) \cdot \frac{t}{0.6} & \text{if } t \leq 0.6 \\
0.70 + (0.40 - 0.70) \cdot \frac{t - 0.6}{0.4} & \text{if } t > 0.6
\end{cases}
\end{align}

\section*{Adaptive Training Updates}
\begin{algorithm}[H]
\caption{Adaptive Update Count}
\KwIn{RL buffer size $|\mathcal{D}_{\text{RL}}|$, Success rate history}
\KwOut{Number of gradient updates $U$}

$U_{\text{base}} \leftarrow 120$ \tcp{Base updates from train.config}

\If{$|\mathcal{D}_{\text{RL}}| < 10000$}{
    $U \leftarrow 40$ \tcp{Early training}
}
\Else{
    $U \leftarrow U_{\text{base}}$
    
    \tcp{Adaptive reduction for stable high performance}
    \If{$|\text{success\_history}| \geq 50$ AND $\text{mean}(\text{success\_history}[-50:]) > 0.6$}{
        $U \leftarrow \max(102, \lfloor 0.85 \cdot U_{\text{base}} \rfloor)$
    }
}
\end{algorithm}

\section*{Terminal Return Calculation}
\begin{align}
R_{\text{shaped}} &= \sum_{t=0}^{T-1} \gamma^t r_t \text{ (standard shaped reward)} \\
R_{\text{terminal}} &= \begin{cases}
R_{\text{success}} & \text{if episode ended with success} \\
R_{\text{collision}} & \text{if episode ended with collision} \\
R_{\text{timeout}} & \text{if episode ended with timeout}
\end{cases} \\
\text{where: } &R_{\text{success}} = 10.0, R_{\text{collision}} = -25.0, R_{\text{timeout}} = 0.0
\end{align}

\section*{Parallel Sampling}
From \texttt{train.config}:
\begin{align}
N_{\text{workers}} &= 4 \text{ (vectorize.num\_workers)} \\
E_{\text{per\_worker}} &= 3 \text{ (vectorize.episodes\_per\_worker)} \\
K_{\text{broadcast}} &= 5 \text{ (vectorize.broadcast\_interval)} \\
\text{Device} &= \text{cuda:0} \text{ (vectorize.worker\_device)}
\end{align}

\section*{Evaluation Protocol}
\begin{algorithm}[H]
\caption{Evaluation Strategy}
\KwIn{Episode number, Evaluation interval = 150}
\KwOut{Validation statistics}

\If{episode $\mod$ 150 == 0}{
    $\text{do\_full} \leftarrow (\text{episode} \mod 300 == 0)$
    
    \If{do\_full}{
        $k \leftarrow |\text{val\_cases}|$ \tcp{Full validation set}
    }
    \Else{
        $k \leftarrow \min(50, |\text{val\_cases}|)$ \tcp{Subset validation}
    }
    
    Run $k$ validation episodes with deterministic policy
    Log validation statistics: success rate, collision rate, navigation time
    
    \tcp{Track best validation performance}
    \If{current performance $>$ best performance}{
        Update best validation record
    }
}
\end{algorithm}

\section*{Model Checkpointing}
\begin{align}
\text{Checkpoint interval: } &500 \text{ episodes} \\
\text{Checkpoint path: } &\text{output\_dir}/\text{rl\_model\_ep}\{episode\}.pth \\
\text{Saved components: } &\{\text{encoder}, \text{actor}, \text{q1}, \text{q2}, \text{obs\_stats}\}
\end{align}

\section*{Training Monitoring}
Logged metrics every episode:
\begin{itemize}
  \item Success rate, collision rate, timeout rate
  \item Average navigation time, total shaped reward
  \item Demo sampling ratio $p_{\text{demo}}$, BC weight $\lambda_{\text{BC}}$
  \item Number of gradient updates, buffer size
  \item Optimizer losses: $\mathcal{L}_q$, $\mathcal{L}_{\text{actor}}$, $\mathcal{L}_{\text{BC}}$, $\mathcal{L}_\alpha$
\end{itemize}

\section*{Final Test Protocol}
\begin{algorithm}[H]
\caption{Final Testing}
\KwIn{Trained policy, Test set size = 300}
\KwOut{Final test statistics}

Set policy to deterministic mode (no exploration)
Run 300 test episodes
Compute final metrics:
\begin{itemize}
  \item Success rate, collision rate, timeout rate
  \item Average navigation time, total reward
\end{itemize}
Report comprehensive summary:
\begin{itemize}
  \item Training performance (last 100 episodes average)
  \item Best validation performance and episode
  \item Final test performance
\end{itemize}
\end{algorithm}

\section*{Configuration Dependencies}
\begin{itemize}
  \item \texttt{train.config}: All training hyperparameters, IL settings, evaluation intervals
  \item \texttt{env.config}: Reward function weights, environment parameters
  \item \texttt{policy.config}: Neural network architecture, action scaling
  \item \texttt{ROBOT\_VPREF\_IDX = 7}: Index of robot's preferred velocity in state vector
  \item \texttt{DEFAULT\_SEED = 42}: Default random seed for reproducibility
\end{itemize}

\end{document}
```