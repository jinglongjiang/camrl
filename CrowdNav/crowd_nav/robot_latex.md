# Robot Agent LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Robot Agent}
\begin{itemize}
  \item $\mathcal{R}$: Robot agent (extends Agent base class)
  \item $\mathbf{s}_R \in \mathbb{R}^9$: Robot observation vector
  \item $\mathbf{s}_H^{(i)} \in \mathbb{R}^5$: Human $i$ observation vector
  \item $\mathcal{H} = \{\mathbf{s}_H^{(1)}, ..., \mathbf{s}_H^{(N)}\}$: Set of human observations
  \item $\pi_R$: Robot policy (neural network or ORCA)
  \item $\mathbf{a}_R$: Robot action
  \item $\mathcal{S}_{\text{joint}}$: Joint state for policy input
\end{itemize}

\section*{Robot State Representation}
\begin{align}
\mathbf{s}_R = [p_x, p_y, g_x, g_y, v_x, v_y, r, v_{\text{pref}}, \theta] \in \mathbb{R}^9
\end{align}

where:
\begin{itemize}
  \item $(p_x, p_y)$: Current position
  \item $(g_x, g_y)$: Goal position  
  \item $(v_x, v_y)$: Current velocity
  \item $r$: Robot radius
  \item $v_{\text{pref}}$: Preferred velocity magnitude
  \item $\theta$: Orientation angle
\end{itemize}

\section*{Robot Configuration}
From \texttt{env.config [robot]} section:
\begin{align}
\text{Visibility: } &\text{visible} = \text{false} \\
\text{Policy: } &\text{policy} = \text{"none"} \text{ (set dynamically)} \\
\text{Radius: } &r = 0.3 \text{ m} \\
\text{Preferred velocity: } &v_{\text{pref}} = 1.0 \text{ m/s} \\
\text{Sensor: } &\text{sensor} = \text{"coordinates"} \\
\text{Kinematics: } &\text{kinematics} = \text{"holonomic"}
\end{align}

\section*{Action Selection Algorithm}
\begin{algorithm}[H]
\caption{Robot Action Selection}
\KwIn{Human observations $\mathcal{H}$, Robot policy $\pi_R$}
\KwOut{Robot action $\mathbf{a}_R$}

\If{$\pi_R$ is None}{
    Raise AttributeError("Policy attribute has to be set!")
}

\tcp{Policy type detection}
$\text{policy\_type} \leftarrow \pi_R.\text{\_\_class\_\_}.\text{\_\_name\_\_}.\text{lower}()$

\If{policy\_type == "orca"}{
    \tcp{ORCA policy uses JointState representation}
    $\text{ob\_list} \leftarrow$ convert $\mathcal{H}$ to ObservableState list
    $\text{joint\_state} \leftarrow$ JointState(get\_full\_state(), ob\_list)
    $\mathbf{a}_R \leftarrow \pi_R.\text{predict}(\text{joint\_state})$
}
\Else{
    \tcp{Neural network policies use float32 array}
    \If{$\mathcal{H}$ is not array}{
        $\text{obs\_array} \leftarrow$ concatenate human observations
        Convert to float32 using to\_array() or numpy conversion
    }
    \Else{
        $\text{obs\_array} \leftarrow \mathcal{H}$ \tcp{Already array format}
    }
    $\mathbf{a}_R \leftarrow \pi_R.\text{predict}(\text{obs\_array})$
}
\end{algorithm}

\section*{Observation Array Generation}
\begin{algorithm}[H]
\caption{Robot Observation Vector}
\KwOut{Robot observation $\mathbf{s}_R \in \mathbb{R}^9$}

$\mathbf{s}_R \leftarrow [p_x, p_y, g_x, g_y, v_x, v_y, r, v_{\text{pref}}, \theta]$

\tcp{Debug output (printed once globally)}
\If{not Robot.\_obs\_shape\_printed}{
    Print('[DEBUG] Robot get\_obs\_array shape:', $\mathbf{s}_R$. shape)
    Robot.\_obs\_shape\_printed $\leftarrow$ True
}

\Return{$\mathbf{s}_R$.astype(np.float32)}
\end{algorithm}

\section*{Policy Interface Adaptation}
The Robot class handles different policy interfaces:

\subsection*{ORCA Policy Interface}
\begin{align}
\text{Input: } &\text{JointState}(\mathcal{S}_{\text{full}}^R, [\mathcal{S}_{\text{obs}}^{H_1}, ..., \mathcal{S}_{\text{obs}}^{H_N}]) \\
\text{Output: } &\text{ActionXY}(v_x, v_y) \\
\text{where: } &\mathcal{S}_{\text{full}}^R = \text{FullState}(p_x, p_y, v_x, v_y, r, g_x, g_y, v_{\text{pref}}, \theta) \\
&\mathcal{S}_{\text{obs}}^{H_i} = \text{ObservableState}(h_{p_x}, h_{p_y}, h_{v_x}, h_{v_y}, h_r)
\end{align}

\subsection*{Neural Network Policy Interface}
\begin{align}
\text{Input: } &\mathbf{s} \in \mathbb{R}^{34} = [\mathbf{s}_R; \mathbf{s}_{H_1}; ...; \mathbf{s}_{H_5}] \\
\text{where: } &\mathbf{s}_R \in \mathbb{R}^9, \mathbf{s}_{H_i} \in \mathbb{R}^5 \\
\text{Output: } &\text{ActionXY}(v_x, v_y)
\end{align}

\section*{Human Observation Processing}
\begin{algorithm}[H]
\caption{Human Observation Concatenation}
\KwIn{Human observations $\mathcal{H}$}
\KwOut{Concatenated array}

\If{$\mathcal{H}$ is list or tuple}{
    concatenated $\leftarrow$ []
    \For{$h$ in $\mathcal{H}$}{
        \If{$h$.has\_attr("to\_array")}{
            array\_h $\leftarrow h$.to\_array().astype(float32)
        }
        \Else{
            array\_h $\leftarrow$ np.array($h$, dtype=float32)
        }
        concatenated.append(array\_h)
    }
    obs\_array $\leftarrow$ np.concatenate(concatenated).astype(float32)
}
\Else{
    obs\_array $\leftarrow \mathcal{H}$ \tcp{Already processed}
}
\end{algorithm}

\section*{Robot Initialization}
\begin{algorithm}[H]
\caption{Robot Setup}
\KwIn{Configuration, Section name}

Call Agent.\_\_init\_\_(config, section) \tcp{Base class initialization}

Initialize observation shape printing flag:
Robot.\_obs\_shape\_printed $\leftarrow$ False \tcp{Class variable}

\tcp{Robot-specific configurations from env.config}
visible $\leftarrow$ False \tcp{Robot is invisible to humans}
policy\_name $\leftarrow$ "none" \tcp{Set dynamically during training}
radius $\leftarrow$ 0.3 m
v\_pref $\leftarrow$ 1.0 m/s
sensor $\leftarrow$ "coordinates"
kinematics $\leftarrow$ "holonomic"
\end{algorithm}

\section*{Policy Assignment}
Robot policies are assigned dynamically during different phases:

\begin{align}
\text{IL Phase: } &\pi_R = \text{ORCA}\_\text{WRAPPER} \text{ (expert demonstrations)} \\
\text{RL Training: } &\pi_R = \text{MambaRL} \text{ (neural network policy)} \\
\text{Evaluation: } &\pi_R = \text{MambaRL} \text{ (deterministic mode)}
\end{align}

\section*{Action Execution}
The robot's action execution follows the Agent base class:

\begin{align}
\text{Position update: } &\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{a}_R \cdot \Delta t \\
\text{Velocity update: } &\mathbf{v}_{t+1} = \mathbf{a}_R \\
\text{Time step: } &\Delta t = 0.25 \text{ s (from env.config)}
\end{align}

\section*{Goal Navigation}
\begin{algorithm}[H]
\caption{Goal Reaching Check}
\KwOut{Boolean indicating goal reached}

$\text{distance} \leftarrow \|\mathbf{p}_R - \mathbf{g}_R\|_2$
\If{distance $< r$}{
    \Return{True} \tcp{Goal reached}
}
\Else{
    \Return{False}
}
\end{algorithm}

\section*{Integration with Training Pipeline}
The Robot class integrates with the training system:

\begin{itemize}
  \item \textbf{Expert Data Collection}: Robot uses ORCA policy for generating demonstrations
  \item \textbf{RL Training}: Robot uses neural network policy (MambaRL) for learning
  \item \textbf{Evaluation}: Robot uses trained policy in deterministic mode
  \item \textbf{Parallel Sampling}: Robot policy broadcast to multiple workers
\end{itemize}

\section*{State Transitions}
During episode execution:

\begin{align}
\mathbf{s}_R(0) &\leftarrow \text{reset to } (0, -4, 0, 4, 0, 0, 0.3, 1.0, \pi/2) \\
\mathbf{s}_R(t+1) &\leftarrow \text{step}(\mathbf{s}_R(t), \mathbf{a}_R(t), \Delta t) \\
\text{Terminal: } &\|\mathbf{p}_R - \mathbf{g}_R\|_2 < r \text{ or collision or timeout}
\end{align}

Initial position: $(0, -4)$, Goal: $(0, 4)$, forming a circle crossing scenario.

\section*{Configuration Dependencies}
\begin{itemize}
  \item \texttt{env.config}: Robot physical parameters and sensor configuration
  \item \texttt{policy.config}: Neural network architecture when using learned policies  
  \item \texttt{train.config}: Policy switching schedule during training phases
  \item Circle radius: 4.0 m (determines initial position and goal)
\end{itemize}

The Robot class provides the interface between the learned policies and the simulation environment, handling different policy types and ensuring consistent action execution.

\end{document}
```