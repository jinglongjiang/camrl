# Agent Base Class LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Agent Base Class}
\begin{itemize}
  \item $\mathbf{p} = (p_x, p_y)$: Agent position
  \item $\mathbf{g} = (g_x, g_y)$: Agent goal position  
  \item $\mathbf{v} = (v_x, v_y)$: Agent velocity
  \item $r$: Agent radius
  \item $v_{\text{pref}}$: Agent preferred velocity
  \item $\theta$: Agent orientation (for non-holonomic kinematics)
  \item $\Delta t$: Time step duration
  \item $\pi$: Agent policy
  \item $\mathcal{S}_{\text{obs}}$: Observable state space
  \item $\mathcal{S}_{\text{full}}$: Full state space
\end{itemize}

\section*{Agent Configuration}
From \texttt{env.config}:
\begin{align}
\text{Visibility: } &\text{visible} \in \{\text{true}, \text{false}\} \\
\text{Preferred velocity: } &v_{\text{pref}} = 1.0 \text{ m/s} \\
\text{Radius: } &r = 0.3 \text{ m} \\
\text{Sensor type: } &\text{sensor} = \text{"coordinates"} \\
\text{Kinematics: } &\text{kinematics} \in \{\text{"holonomic"}, \text{"differential"}\}
\end{align}

\section*{State Representations}
\subsection*{Observable State}
\begin{align}
\mathcal{S}_{\text{obs}} = \{p_x, p_y, v_x, v_y, r\} \in \mathbb{R}^5
\end{align}

\subsection*{Full State}  
\begin{align}
\mathcal{S}_{\text{full}} = \{p_x, p_y, v_x, v_y, r, g_x, g_y, v_{\text{pref}}, \theta\} \in \mathbb{R}^9
\end{align}

\section*{Agent Initialization and Configuration}
\begin{algorithm}[H]
\caption{Agent Initialization}
\KwIn{Configuration file, Section name}
\KwOut{Configured agent}

Read configuration parameters:
\begin{itemize}
    \item $\text{visible} \leftarrow$ config.getboolean(section, 'visible')
    \item $v_{\text{pref}} \leftarrow$ config.getfloat(section, 'v\_pref') 
    \item $r \leftarrow$ config.getfloat(section, 'radius')
    \item $\text{policy\_name} \leftarrow$ config.get(section, 'policy')
    \item $\text{sensor} \leftarrow$ config.get(section, 'sensor')
\end{itemize}

$\pi \leftarrow$ policy\_factory[policy\_name]() \tcp{Instantiate policy}
$\text{kinematics} \leftarrow \pi.\text{kinematics}$ \tcp{Get kinematics from policy}

Initialize state variables:
$\mathbf{p}, \mathbf{g}, \mathbf{v}, \theta \leftarrow \text{None}$
$\Delta t \leftarrow \text{None}$
\end{algorithm}

\section*{State Management}
\begin{algorithm}[H]
\caption{Agent State Setting}
\KwIn{Position $(p_x, p_y)$, Goal $(g_x, g_y)$, Velocity $(v_x, v_y)$, Orientation $\theta$}

$\mathbf{p} \leftarrow (p_x, p_y)$
$\mathbf{g} \leftarrow (g_x, g_y)$ 
$\mathbf{v} \leftarrow (v_x, v_y)$
$\theta \leftarrow \theta$

\If{radius provided}{
    $r \leftarrow \text{radius}$
}
\If{preferred velocity provided}{
    $v_{\text{pref}} \leftarrow \text{v\_pref}$
}
\end{algorithm}

\section*{Observation Generation}
\begin{algorithm}[H]
\caption{Observable State Generation}
\KwOut{Observable state array $\mathbf{s}_{\text{obs}} \in \mathbb{R}^5$}

$\mathbf{s}_{\text{obs}} \leftarrow [p_x, p_y, v_x, v_y, r]$
\Return{$\mathbf{s}_{\text{obs}}$.astype(np.float32)}
\end{algorithm}

\section*{Next State Prediction}
\begin{algorithm}[H]
\caption{Next Observable State Prediction}
\KwIn{Action $\mathbf{a}$, Time step $\Delta t$}
\KwOut{Predicted next observable state}

Validate action compatibility with kinematics
$\mathbf{p}_{\text{next}} \leftarrow$ compute\_position($\mathbf{a}$, $\Delta t$)

\If{kinematics == 'holonomic'}{
    $\mathbf{v}_{\text{next}} \leftarrow (a_{v_x}, a_{v_y})$
}
\Else{
    $\theta_{\text{next}} \leftarrow \theta + a_r$
    $\mathbf{v}_{\text{next}} \leftarrow (a_v \cos \theta_{\text{next}}, a_v \sin \theta_{\text{next}})$
}

$\mathbf{s}_{\text{obs,next}} \leftarrow [p_{\text{next},x}, p_{\text{next},y}, v_{\text{next},x}, v_{\text{next},y}, r]$
\Return{$\mathbf{s}_{\text{obs,next}}$}
\end{algorithm}

\section*{Position Computation}
\subsection*{Holonomic Kinematics}
\begin{align}
\mathbf{p}_{\text{next}} &= \mathbf{p} + \mathbf{v}_{\text{action}} \cdot \Delta t \\
&= (p_x + a_{v_x} \cdot \Delta t, p_y + a_{v_y} \cdot \Delta t)
\end{align}

\subsection*{Differential Drive Kinematics}
\begin{align}
\theta_{\text{next}} &= \theta + a_r \\
\mathbf{p}_{\text{next}} &= \mathbf{p} + a_v \cdot (\cos \theta_{\text{next}}, \sin \theta_{\text{next}}) \cdot \Delta t \\
&= (p_x + a_v \cos(\theta + a_r) \cdot \Delta t, p_y + a_v \sin(\theta + a_r) \cdot \Delta t)
\end{align}

\section*{Agent Step Update}
\begin{algorithm}[H]
\caption{Agent State Update}
\KwIn{Action $\mathbf{a}$}

Validate action compatibility
$\mathbf{p}_{\text{new}} \leftarrow$ compute\_position($\mathbf{a}$, $\Delta t$)
$\mathbf{p} \leftarrow \mathbf{p}_{\text{new}}$

\If{kinematics == 'holonomic'}{
    $\mathbf{v} \leftarrow (a_{v_x}, a_{v_y})$
}
\Else{
    $\theta \leftarrow (\theta + a_r) \bmod 2\pi$ \tcp{Wrap angle}
    $\mathbf{v} \leftarrow (a_v \cos \theta, a_v \sin \theta)$
}
\end{algorithm}

\section*{Goal Reaching Detection}
\begin{align}
\text{reached\_destination} &= \|\mathbf{p} - \mathbf{g}\|_2 < r \\
&= \sqrt{(p_x - g_x)^2 + (p_y - g_y)^2} < r
\end{align}

\section*{Random Attribute Sampling}
For training diversity:
\begin{align}
v_{\text{pref}} &\sim \text{Uniform}(0.5, 1.5) \text{ m/s} \\
r &\sim \text{Uniform}(0.3, 0.5) \text{ m}
\end{align}

\section*{Action Validation}
\begin{algorithm}[H]
\caption{Action Type Validation}
\KwIn{Action $\mathbf{a}$, Kinematics type}

\If{kinematics == 'holonomic'}{
    Assert $\mathbf{a}$ is ActionXY type
    Verify $\mathbf{a}$ has attributes: $v_x$, $v_y$
}
\Else{
    Assert $\mathbf{a}$ is ActionRot type  
    Verify $\mathbf{a}$ has attributes: $v$, $r$ (linear and angular velocity)
}
\end{algorithm}

\section*{State Access Methods}
\begin{align}
\text{get\_position}() &\rightarrow (p_x, p_y) \\
\text{get\_goal\_position}() &\rightarrow (g_x, g_y) \\
\text{get\_velocity}() &\rightarrow (v_x, v_y) \\
\text{get\_observable\_state}() &\rightarrow \text{ObservableState}(p_x, p_y, v_x, v_y, r) \\
\text{get\_full\_state}() &\rightarrow \text{FullState}(p_x, p_y, v_x, v_y, r, g_x, g_y, v_{\text{pref}}, \theta) \\
\text{get\_obs\_array}() &\rightarrow [p_x, p_y, v_x, v_y, r] \in \mathbb{R}^5
\end{align}

\section*{Policy Integration}
\begin{algorithm}[H]
\caption{Policy Management}
\KwIn{New policy $\pi_{\text{new}}$}

$\pi \leftarrow \pi_{\text{new}}$
$\text{kinematics} \leftarrow \pi.\text{kinematics}$ \tcp{Update kinematics constraint}

Set policy parameters:
$\pi.\text{time\_step} \leftarrow \Delta t$
\end{algorithm}

\section*{Configuration Parameters}
From \texttt{env.config}:
\begin{itemize}
  \item \textbf{Robot section}: visible=false, policy=none, radius=0.3, v\_pref=1, sensor=coordinates
  \item \textbf{Humans section}: visible=true, policy=orca, radius=0.3, v\_pref=1, sensor=coordinates  
  \item \textbf{Action space}: kinematics=holonomic, time\_step=0.25
\end{itemize}

The Agent class serves as the base class for both Robot and Human agents, providing:
\begin{itemize}
  \item Unified state management and representation
  \item Kinematics-aware motion computation
  \item Policy integration interface
  \item Observation generation for RL algorithms
  \item Goal-directed behavior primitives
\end{itemize}

\end{document}
```