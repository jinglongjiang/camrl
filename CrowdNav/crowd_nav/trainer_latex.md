# SAC Trainer LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - SAC Trainer}
\begin{itemize}
  \item $\theta$: Actor network parameters
  \item $\phi_1, \phi_2$: Critic network parameters (twin Q-networks)
  \item $\bar{\phi}_1, \bar{\phi}_2$: Target critic network parameters
  \item $\alpha$: Entropy temperature coefficient
  \item $\gamma = 0.95$: Discount factor (from train.config)
  \item $\tau = 0.0075$: Polyak averaging coefficient (from train.config)
  \item $B = 1536$: Batch size (from train.config)
  \item $H_{\text{target}} = -2.4$: Target entropy (from train.config)
  \item $\lambda_{\text{BC}}$: Behavioral cloning regularization weight
  \item $p_{\text{demo}}$: Demonstration data sampling ratio
  \item $\beta_{\text{AWBC}} = 2.5$: AWBC temperature parameter (from train.config)
  \item $\mathcal{D}_{\text{RL}}$: RL replay buffer
  \item $\mathcal{D}_E$: Expert demonstration buffer
\end{itemize}

\section*{Action Normalization}
\begin{align}
\mathbf{a}_{\text{phys}} &\in \mathbb{R}^2 \text{ (physical velocities from buffer)} \\
v_{\text{pref}} &= s[\text{v\_pref\_idx}] = s[7] \text{ (robot preferred velocity)} \\
\sigma_{\text{scale}} &= 0.65 \text{ (from policy.config)} \\
\mathbf{a}_{\text{norm}} &= \text{clamp}\left(\frac{\mathbf{a}_{\text{phys}}}{v_{\text{pref}} \cdot \sigma_{\text{scale}}}, -1, 1\right)
\end{align}

\section*{Priority Sampling Strategy}
\begin{algorithm}[H]
\caption{Enhanced Priority Sampling}
\KwIn{RL buffer $\mathcal{D}_{\text{RL}}$, Batch size $B$}
\KwOut{Sampled batch with priority bias}

$n_{\text{priority}} \leftarrow \lfloor 0.35 \cdot B \rfloor$ \tcp{35% priority samples (updated from 20%)}
$n_{\text{random}} \leftarrow B - n_{\text{priority}}$

\tcp{Base random sampling}
$\{(\mathbf{s}_i, \mathbf{a}_i, r_i, \mathbf{s}'_i, d_i)\}_{i=1}^{n_{\text{random}}} \leftarrow \text{RandomSample}(\mathcal{D}_{\text{RL}})$

\tcp{Priority sampling from high-reward trajectories}
\If{$n_{\text{priority}} > 0$ AND $|\mathcal{D}_{\text{RL}}| \geq 128$}{
    $\text{probe\_size} \leftarrow \min(2048, |\mathcal{D}_{\text{RL}}|)$
    Sample probe indices uniformly
    $\{r_j\}_{j=1}^{\text{probe\_size}} \leftarrow$ rewards from probe
    
    \tcp{Prioritize success trajectories}
    $\mathcal{S}_{\text{success}} \leftarrow \{j : r_j > 5.0\}$ \tcp{Success threshold}
    
    \If{$|\mathcal{S}_{\text{success}}| > n_{\text{priority}}/2$}{
        $\text{candidates} \leftarrow \mathcal{S}_{\text{success}}$
    }
    \Else{
        $\text{threshold} \leftarrow \text{percentile}(\{r_j\}, 70)$
        $\text{candidates} \leftarrow \{j : r_j \geq \max(\text{threshold}, -5.0)\}$
    }
    
    Sample $n_{\text{priority}}$ transitions from candidates
    Concatenate with random samples
}
\end{algorithm}

\section*{SAC Loss Functions}
\subsection*{Critic Loss (Twin Q-Networks)}
\begin{align}
\text{Target computation: } y &= r + \gamma (1-d) \left[\min(Q_{\bar{\phi}_1}(\mathbf{s}', \tilde{\mathbf{a}}'), Q_{\bar{\phi}_2}(\mathbf{s}', \tilde{\mathbf{a}}')) - \alpha \log \pi_\theta(\tilde{\mathbf{a}}'|\mathbf{s}')\right] \\
\text{where: } \tilde{\mathbf{a}}' &\sim \pi_\theta(\cdot|\mathbf{s}') \\
\mathcal{L}_Q &= \text{HuberLoss}(Q_{\phi_1}(\mathbf{s}, \mathbf{a}), y) + \text{HuberLoss}(Q_{\phi_2}(\mathbf{s}, \mathbf{a}), y) \\
\text{HuberLoss}(x, y) &= \begin{cases}
\frac{1}{2}(x-y)^2 & \text{if } |x-y| \leq \delta \\
\delta(|x-y| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
\end{align}
where $\delta = 1.0$ for numerical stability.

\subsection*{Actor Loss (SAC + AWBC)}
\begin{align}
\mathcal{L}_{\text{SAC}} &= \mathbb{E}_{(\mathbf{s}, \tilde{\mathbf{a}}) \sim \mathcal{D}_{\text{RL}}} \left[ \alpha \log \pi_\theta(\tilde{\mathbf{a}}|\mathbf{s}) - \min(Q_{\phi_1}(\mathbf{s}, \tilde{\mathbf{a}}), Q_{\phi_2}(\mathbf{s}, \tilde{\mathbf{a}})) \right] \\
\text{where: } \tilde{\mathbf{a}} &\sim \pi_\theta(\cdot|\mathbf{s}) \\
\mathcal{L}_{\text{AWBC}} &= \mathbb{E}_{(\mathbf{s}, \mathbf{a}^*) \sim \mathcal{D}_E} \left[ w(\mathbf{s}, \mathbf{a}^*) \cdot \|\pi_\theta(\mathbf{s}) - \mathbf{a}^*\|_2^2 \right] \\
\mathcal{L}_{\text{Actor}} &= \mathcal{L}_{\text{SAC}} + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{AWBC}}
\end{align}

\section*{Advantage-Weighted Behavioral Cloning (AWBC)}
\begin{align}
\text{Advantage: } A(\mathbf{s}, \mathbf{a}^*) &= Q(\mathbf{s}, \mathbf{a}^*) - Q(\mathbf{s}, \tilde{\mathbf{a}}) \\
\text{where: } \tilde{\mathbf{a}} &\sim \pi_\theta(\cdot|\mathbf{s}) \\
\text{Weight: } w(\mathbf{s}, \mathbf{a}^*) &= \sigma(\beta_{\text{AWBC}} \cdot A(\mathbf{s}, \mathbf{a}^*)) \\
&= \frac{1}{1 + \exp(-\beta_{\text{AWBC}} \cdot A(\mathbf{s}, \mathbf{a}^*))} \\
\text{where: } \beta_{\text{AWBC}} &= 2.5 \text{ (temperature parameter)}
\end{align}

The AWBC weight $w \in (0,1)$ automatically emphasizes expert actions that have higher Q-values than the current policy's actions.

\section*{Entropy Temperature Update}
\begin{align}
\mathcal{L}_\alpha &= \mathbb{E}_{\mathbf{s} \sim \mathcal{D}_{\text{RL}}} \left[ -\alpha \left( \log \pi_\theta(\tilde{\mathbf{a}}|\mathbf{s}) + H_{\text{target}} \right) \right] \\
\text{where: } \tilde{\mathbf{a}} &\sim \pi_\theta(\cdot|\mathbf{s}) \\
\alpha &= \exp(\log \alpha) \text{ (ensure positivity)}
\end{align}

\section*{Target Network Updates (Polyak Averaging)}
\begin{align}
\bar{\phi}_1 &\leftarrow (1-\tau) \bar{\phi}_1 + \tau \phi_1 \\
\bar{\phi}_2 &\leftarrow (1-\tau) \bar{\phi}_2 + \tau \phi_2 \\
\text{where: } \tau &= 0.0075 \text{ (from train.config)}
\end{align}

\section*{Complete Training Algorithm}
\begin{algorithm}[H]
\caption{SAC with AWBC Training Step}
\KwIn{Buffers $\mathcal{D}_{\text{RL}}, \mathcal{D}_E$, Schedules $p_{\text{demo}}, \lambda_{\text{BC}}$, Updates $U$}
\KwOut{Training metrics}

\For{$u = 1$ to $U$}{
    $B_{\text{demo}} \leftarrow \min(\lfloor B \cdot p_{\text{demo}} \rfloor, 128)$
    $B_{\text{RL}} \leftarrow B - B_{\text{demo}}$
    
    \tcp{Sample batches}
    $\{(\mathbf{s}, \mathbf{a}, r, \mathbf{s}', d)\} \leftarrow \text{PrioritySample}(\mathcal{D}_{\text{RL}}, B_{\text{RL}})$
    $\{(\mathbf{s}_E, \mathbf{a}_E^*)\} \leftarrow \text{RandomSample}(\mathcal{D}_E, B_{\text{demo}})$
    
    \tcp{Normalize actions and observations}
    $\mathbf{a} \leftarrow \text{normalize\_actions}(\mathbf{s}, \mathbf{a})$
    $\hat{\mathbf{s}}, \hat{\mathbf{s}}' \leftarrow \text{normalize\_obs}(\mathbf{s}), \text{normalize\_obs}(\mathbf{s}')$
    $\mathbf{a}_E \leftarrow \text{normalize\_actions}(\mathbf{s}_E, \mathbf{a}_E^*)$
    
    \tcp{Update Critics}
    $\alpha \leftarrow \exp(\log \alpha)$ \tcp{Current temperature}
    Compute target $y$ using target networks
    $\mathcal{L}_Q \leftarrow \text{HuberLoss}(Q_{\phi_1}, y) + \text{HuberLoss}(Q_{\phi_2}, y)$
    Update $\phi_1, \phi_2$ using $\nabla_{\phi_1, \phi_2} \mathcal{L}_Q$
    
    \tcp{Update Actor}
    $\mathcal{L}_{\text{SAC}} \leftarrow$ SAC actor loss
    $\mathcal{L}_{\text{AWBC}} \leftarrow$ AWBC loss with advantage weighting
    $\mathcal{L}_{\text{Actor}} \leftarrow \mathcal{L}_{\text{SAC}} + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{AWBC}}$
    Update $\theta$ using $\nabla_\theta \mathcal{L}_{\text{Actor}}$
    
    \tcp{Update Temperature}
    $\mathcal{L}_\alpha \leftarrow$ entropy temperature loss
    Update $\log \alpha$ using $\nabla_{\log \alpha} \mathcal{L}_\alpha$
    
    \tcp{Update Target Networks}
    $\bar{\phi}_1 \leftarrow (1-\tau)\bar{\phi}_1 + \tau\phi_1$
    $\bar{\phi}_2 \leftarrow (1-\tau)\bar{\phi}_2 + \tau\phi_2$
}
\end{algorithm}

\section*{Automatic Mixed Precision (AMP)}
\begin{align}
\text{Forward pass: } &\text{autocast}(\text{enabled} = \text{use\_amp}) \\
\text{Backward pass: } &\text{scaler.scale}(\mathcal{L}).backward() \\
\text{Gradient clipping: } &\text{scaler.unscale\_}(\text{optimizer}) \\
&\text{clip\_grad\_norm\_}(\text{parameters}, 1.0) \\
\text{Optimizer step: } &\text{scaler.step}(\text{optimizer}) \\
&\text{scaler.update}()
\end{align}

AMP is enabled only on CUDA devices and provides significant speedup for large models.

\section*{Memory Optimization}
\begin{align}
\text{Pinned memory transfer: } &x.\text{pin\_memory}().\text{to}(\text{device}, \text{non\_blocking=True}) \\
\text{Gradient accumulation: } &\text{optimizer.zero\_grad}(\text{set\_to\_None=True}) \\
\text{Detached encoding: } &\text{encode}(\mathbf{s}.\text{detach}()) \text{ for critic update}
\end{align}

\section*{Configuration Parameters}
From \texttt{train.config}:
\begin{itemize}
  \item \texttt{gamma = 0.95}: Discount factor $\gamma$
  \item \texttt{tau = 0.0075}: Polyak averaging coefficient $\tau$
  \item \texttt{lr\_actor = 2e-4}: Actor learning rate $\alpha_{\text{actor}}$
  \item \texttt{lr\_critic = 2.5e-4}: Critic learning rate $\alpha_{\text{critic}}$
  \item \texttt{lr\_alpha = 2.4e-4}: Temperature learning rate $\alpha_{\alpha}$
  \item \texttt{batch\_size = 1536}: Training batch size $B$
  \item \texttt{target\_entropy = -2.4}: Target entropy $H_{\text{target}}$
  \item \texttt{grad\_clip = 1.0}: Gradient clipping threshold
  \item \texttt{awbc\_beta = 2.5}: AWBC temperature $\beta_{\text{AWBC}}$
  \item \texttt{use\_amp = True}: Enable automatic mixed precision
\end{itemize}

\section*{Return Metrics}
The trainer returns the following metrics per training step:
\begin{itemize}
  \item \texttt{loss\_q}: Average critic loss $\mathcal{L}_Q$
  \item \texttt{loss\_actor}: Average actor loss $\mathcal{L}_{\text{Actor}}$
  \item \texttt{bc\_loss}: Average BC loss $\mathcal{L}_{\text{AWBC}}$
  \item \texttt{alpha}: Current entropy temperature $\alpha$
  \item \texttt{awbc\_w}: Average AWBC weight $\bar{w}$
\end{itemize}

\end{document}
```