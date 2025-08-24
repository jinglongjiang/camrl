# Configuration System LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Configuration System}
\begin{itemize}
  \item $T = 100$: Episode time limit (seconds)
  \item $\Delta t = 0.25$: Simulation time step (seconds) 
  \item $N_H = 5$: Maximum number of humans
  \item $R_{\text{success}} = 10.0$: Success reward
  \item $R_{\text{collision}} = -25.0$: Collision penalty
  \item $\gamma = 0.95$: RL discount factor
  \item $\alpha_{\text{lr}}$: Learning rate parameters
  \item $B = 1536$: Training batch size
  \item $H = 64$: Neural network hidden dimension
  \item $K = 3$: Top-K human selection
  \item $\sigma_{\text{scale}} = 0.65$: Action scaling factor
\end{itemize}

\section*{Environment Configuration (env.config)}

\subsection*{Simulation Parameters}
\begin{align}
\text{Time limit: } &T = 100 \text{ seconds} \\
\text{Time step: } &\Delta t = 0.25 \text{ seconds} \\
\text{Validation episodes: } &N_{\text{val}} = 100 \\
\text{Test episodes: } &N_{\text{test}} = 300 \\
\text{Randomize attributes: } &\text{True}
\end{align}

\subsection*{Reward Function Parameters}
\begin{align}
R_{\text{success}} &= 10.0 \text{ (terminal success reward)} \\
R_{\text{collision}} &= -25.0 \text{ (terminal collision penalty)} \\
R_{\text{timeout}} &= 0.0 \text{ (terminal timeout penalty)} \\
d_{\text{discomfort}} &= 0.35 \text{ m (social discomfort threshold)} \\
\text{penalty\_factor} &= 10.0 \text{ (discomfort scaling)}
\end{align}

\subsection*{Reward Shaping Weights}
\begin{align}
w_{\text{prog}} &= 0.8 \text{ (progress toward goal)} \\
w_{\text{goal}} &= 3.0 \text{ (goal achievement emphasis)} \\
w_{\text{coll}} &= 12.0 \text{ (collision avoidance)} \\
w_{\text{soc}} &= 5.0 \text{ (social compliance)} \\
w_{\text{time}} &= 0.01 \text{ (time cost per step)} \\
w_{\text{ttc}} &= 6.0 \text{ (time-to-collision penalty)} \\
w_{\text{speed}} &= 0.25 \text{ (speed regulation)} \\
w_{\text{relv}} &= 2.5 \text{ (relative velocity penalty)}
\end{align}

\subsection*{Advanced Reward Components}
\begin{align}
\text{TTC threshold: } &\tau_{\text{TTC}} = 4.0 \text{ seconds} \\
\text{Speed target: } &v_{\text{target}} = 0.4 \times v_{\text{pref}} \\
\text{No-progress threshold: } &\epsilon_{\text{prog}} = 0.015 \text{ m} \\
\text{No-progress patience: } &n_{\text{patience}} = 24 \text{ steps}
\end{align}

\subsection*{Scenario Configuration}
\begin{align}
\text{Training scenario: } &\text{"circle\_crossing"} \\
\text{Test scenario: } &\text{"circle\_crossing"} \\
\text{Circle radius: } &R_{\text{circle}} = 4.0 \text{ m} \\
\text{Square width: } &W_{\text{square}} = 10.0 \text{ m} \\
\text{Number of humans: } &N_H = 5
\end{align}

\subsection*{Agent Properties}
\subsubsection*{Robot Configuration}
\begin{align}
\text{Visibility: } &\text{False} \\
\text{Policy: } &\text{"none"} \text{ (set dynamically)} \\
\text{Radius: } &r_R = 0.3 \text{ m} \\
\text{Preferred velocity: } &v_{\text{pref},R} = 1.0 \text{ m/s} \\
\text{Sensor: } &\text{"coordinates"}
\end{align}

\subsubsection*{Human Configuration}
\begin{align}
\text{Visibility: } &\text{True} \\
\text{Policy: } &\text{"orca"} \\
\text{Radius: } &r_H = 0.3 \text{ m} \\
\text{Preferred velocity: } &v_{\text{pref},H} = 1.0 \text{ m/s} \\
\text{Sensor: } &\text{"coordinates"}
\end{align}

\subsection*{ORCA Expert Parameters}
\begin{align}
\text{Neighbor distance: } &d_{\text{neighbor}} = 3.0 \text{ m} \\
\text{Max neighbors: } &N_{\text{neighbor}} = 8 \\
\text{Time horizon: } &\tau_{\text{horizon}} = 4.0 \text{ s} \\
\text{Obstacle horizon: } &\tau_{\text{obs}} = 3.0 \text{ s} \\
\text{Safety space: } &s_{\text{safety}} = 0.05 \text{ m} \\
\text{Label inflation: } &s_{\text{label}} = 0.10 \text{ m} \\
\text{TTC brake threshold: } &\tau_{\text{brake}} = 2.0 \text{ s} \\
\text{Minimum speed ratio: } &\rho_{\text{min}} = 0.3
\end{align}

\section*{Training Configuration (train.config)}

\subsection*{Episode Structure}
\begin{align}
\text{Training episodes: } &E_{\text{train}} = \text{varies by experiment} \\
\text{Sample episodes: } &E_{\text{sample}} = 12 \text{ per training step} \\
\text{Evaluation interval: } &\Delta E_{\text{eval}} = 150 \text{ episodes} \\
\text{Checkpoint interval: } &\Delta E_{\text{save}} = 500 \text{ episodes} \\
\text{Buffer capacity: } &C = 100000 \text{ transitions}
\end{align}

\subsection*{Exploration Schedule}
\begin{align}
\epsilon_{\text{start}} &= 0.08 \\
\epsilon_{\text{end}} &= 0.03 \\
\epsilon_{\text{decay}} &= 1500 \text{ episodes} \\
\epsilon(t) &= \begin{cases}
\epsilon_{\text{start}} + (\epsilon_{\text{end}} - \epsilon_{\text{start}}) \frac{t}{\epsilon_{\text{decay}}} & t \leq \epsilon_{\text{decay}} \\
\epsilon_{\text{end}} & t > \epsilon_{\text{decay}}
\end{cases}
\end{align}

\subsection*{Evaluation Strategy}
\begin{align}
\text{Validation subset: } &N_{\text{val,sub}} = 50 \text{ episodes} \\
\text{Full evaluation every: } &\Delta E_{\text{full}} = 300 \text{ episodes} \\
\text{Training statistics stride: } &s_{\text{stat}} = 5 \text{ episodes}
\end{align}

\subsection*{SAC Hyperparameters}
\begin{align}
\gamma &= 0.95 \text{ (discount factor)} \\
\tau &= 0.0075 \text{ (Polyak averaging)} \\
\alpha_{\text{actor}} &= 2 \times 10^{-4} \text{ (actor learning rate)} \\
\alpha_{\text{critic}} &= 2.5 \times 10^{-4} \text{ (critic learning rate)} \\
\alpha_{\alpha} &= 2.4 \times 10^{-4} \text{ (temperature learning rate)} \\
B &= 1536 \text{ (batch size)} \\
H_{\text{target}} &= -2.4 \text{ (target entropy)} \\
\text{AMP} &= \text{True} \text{ (automatic mixed precision)} \\
\text{grad\_clip} &= 1.0 \text{ (gradient clipping)} \\
\beta_{\text{AWBC}} &= 2.5 \text{ (advantage weighting)}
\end{align}

\subsection*{Imitation Learning}
\begin{align}
E_{\text{IL}} &= 150 \text{ (expert episodes)} \\
N_{\text{BC}} &= 3000 \text{ (BC pretraining epochs)} \\
\alpha_{\text{BC}} &= 3 \times 10^{-4} \text{ (BC learning rate)} \\
\pi_{\text{expert}} &= \text{"orca"} \text{ (expert policy)}
\end{align}

\subsection*{Parallel Sampling}
\begin{align}
\text{Enable vectorization: } &\text{True} \\
\text{Number of workers: } &N_{\text{workers}} = 4 \\
\text{Episodes per worker: } &E_{\text{worker}} = 3 \\
\text{Broadcast interval: } &K_{\text{broadcast}} = 5 \\
\text{Worker device: } &\text{"cuda:0"}
\end{align}

\section*{Policy Configuration (policy.config)}

\subsection*{Mamba Architecture}
\begin{align}
\text{Hidden dimension: } &H = 64 \\
\text{Number of blocks: } &L = 4 \\
\text{State dimension: } &D_{\text{state}} = 34 \\
\text{Convolution dimension: } &D_{\text{conv}} = 4 \\
\text{Expansion factor: } &\text{expand} = 2 \\
\text{Robot features: } &D_R = 9 \\
\text{Human features: } &D_H = 5 \\
\text{Human count: } &N_H = 5 \\
\text{Top-K selection: } &K = 3 \\
\text{Dropout: } &p_{\text{dropout}} = 0.0
\end{align}

\subsection*{Policy Parameters}
\begin{align}
\text{Discount factor: } &\gamma = 0.95 \\
\text{Action scaling: } &\sigma_{\text{scale}} = 0.65 \\
\text{Multiagent training: } &\text{True} \\
\text{Environment stochastic: } &\text{True}
\end{align}

\subsection*{SARL Configuration}
\begin{align}
\text{Hidden dimensions: } &[150, 100, 100, 1] \\
\text{Attention heads: } &N_{\text{heads}} = 4 \\
\text{Attention dimension: } &D_{\text{attn}} = 16
\end{align}

\section*{Configuration Integration}

\subsection*{Parameter Flow}
\begin{algorithm}[H]
\caption{Configuration Parameter Usage}

\tcp{Environment setup}
env $\leftarrow$ CrowdSim()
env.configure(env\_config) \tcp{Reward weights, scenario parameters}

\tcp{Policy creation}  
policy $\leftarrow$ policy\_factory[policy\_name](policy\_config)
policy.configure(policy\_config) \tcp{Network architecture}

\tcp{Training setup}
trainer $\leftarrow$ Trainer(policy, train\_config) \tcp{SAC parameters}
explorer $\leftarrow$ Explorer(env, train\_config) \tcp{Sampling parameters}

\tcp{Expert policy for IL}
expert $\leftarrow$ policy\_factory[expert\_name](policy\_config) 
expert.configure(env\_config) \tcp{ORCA parameters from env.config}
\end{algorithm}

\subsection*{Parameter Consistency}
Critical parameter alignments across configs:
\begin{align}
\text{Action scaling: } &\sigma_{\text{scale}} \text{ (policy.config)} \rightarrow \text{trainer normalization} \\
\text{Robot v\_pref index: } &\text{ROBOT\_VPREF\_IDX} = 7 \text{ (hardcoded)} \\
\text{State dimensions: } &D_{\text{state}} = 9 + 5 \times N_H = 34 \\
\text{Time step: } &\Delta t = 0.25 \text{ (env.config)} \rightarrow \text{all components} \\
\text{Discount factor: } &\gamma = 0.95 \text{ (consistent across train/policy configs)}
\end{align}

\subsection*{Dynamic Parameter Updates}
Some parameters are updated during training:
\begin{align}
\text{Learning rates: } &\alpha(t) = \alpha_0 \times \text{schedule}(t) \\
\text{Target entropy: } &H_{\text{target}}(t) = -2.2 - 0.6 \times \min(t/0.8, 1) \\
\text{Demo ratio: } &p_{\text{demo}}(t) = 0.50 - 0.15 \times t \\
\text{BC weight: } &\lambda_{\text{BC}}(t) = \text{piecewise\_schedule}(t)
\end{align}

\section*{Configuration Validation}
\begin{algorithm}[H]
\caption{Configuration Consistency Checks}

\tcp{Dimension consistency}
Assert $D_{\text{state}} = D_R + N_H \times D_H$
Assert ROBOT\_VPREF\_IDX $< D_R$ 

\tcp{Reward weight sanity}
Assert $w_{\text{coll}} > w_{\text{prog}}$ \tcp{Safety prioritization}
Assert $R_{\text{collision}} < 0$ and $R_{\text{success}} > 0$

\tcp{Training parameter bounds}
Assert $0 < \gamma < 1$ \tcp{Discount factor bounds}
Assert $\tau > 0$ \tcp{Polyak coefficient}
Assert $B > 0$ \tcp{Batch size}

\tcp{Architecture constraints}
Assert $K \leq N_H$ \tcp{Top-K selection}
Assert $H > 0$ \tcp{Hidden dimension}
\end{algorithm}

The configuration system ensures consistent parameter usage across all components while maintaining flexibility for hyperparameter tuning and architectural modifications.

\end{document}
```