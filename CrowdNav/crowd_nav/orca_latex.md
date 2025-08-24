# ORCA Expert Policy LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - ORCA Policy}
\begin{itemize}
  \item $\mathcal{A}_i$: Agent $i$ with position $\mathbf{p}_i$, velocity $\mathbf{v}_i$, radius $r_i$
  \item $\mathbf{v}_i^{\text{pref}}$: Preferred velocity for agent $i$
  \item $v_i^{\max}$: Maximum speed for agent $i$
  \item $\tau$: Time horizon for collision avoidance
  \item $\tau_{\text{obs}}$: Time horizon for obstacle avoidance
  \item $d_{\text{neighbor}}$: Neighbor detection distance
  \item $N_{\max}$: Maximum number of neighbors to consider
  \item $\text{TTC}$: Time-to-collision
  \item $\mathbf{u}_i^{\text{new}}$: New velocity computed by ORCA
\end{itemize}

\section*{ORCA Configuration Parameters}
From \texttt{env.config [orca]} section:
\begin{align}
\text{Neighbor distance: } &d_{\text{neighbor}} = 3.0 \text{ m} \\
\text{Max neighbors: } &N_{\max} = 8 \\
\text{Time horizon: } &\tau = 4.0 \text{ s} \\
\text{Obstacle horizon: } &\tau_{\text{obs}} = 3.0 \text{ s} \\
\text{Agent radius: } &r = 0.3 \text{ m} \\
\text{Max speed: } &v^{\max} = 1.2 \text{ m/s} \\
\text{Safety space: } &s_{\text{safety}} = 0.05 \text{ m} \\
\text{Label inflation: } &s_{\text{label}} = 0.10 \text{ m} \\
\text{Deceleration factor: } &k_{\text{slow}} = 2.0 \\
\text{Symmetry breaking: } &\epsilon_{\text{noise}} = 0.03 \\
\text{TTC brake threshold: } &\text{TTC}_{\text{brake}} = 2.0 \text{ s} \\
\text{Minimum speed ratio: } &\rho_{\text{min}} = 0.3
\end{align}

\section*{Preferred Velocity Calculation}
\begin{algorithm}[H]
\caption{Goal-Directed Preferred Velocity}
\KwIn{Current position $\mathbf{p} = (p_x, p_y)$, Goal $\mathbf{g} = (g_x, g_y)$, Preferred speed $v_{\text{pref}}$, Radius $r$}
\KwOut{Preferred velocity $\mathbf{v}^{\text{pref}}$}

$\mathbf{d} \leftarrow \mathbf{g} - \mathbf{p}$ \tcp{Direction to goal}
$\text{dist} \leftarrow \|\mathbf{d}\|_2$ \tcp{Distance to goal}

\If{$\text{dist} < 10^{-8}$}{
    \Return{$(0, 0)$} \tcp{Already at goal}
}

$\hat{\mathbf{d}} \leftarrow \mathbf{d} / \text{dist}$ \tcp{Unit direction vector}

\tcp{Speed limits}
$v_{\text{limit}} \leftarrow \min\{v_{\text{pref}}, v^{\max}, \text{dist}/\Delta t\}$

\tcp{Deceleration near goal}
\If{$\text{dist} < k_{\text{slow}} \cdot r$}{
    $v_{\text{limit}} \leftarrow v_{\text{limit}} \cdot \frac{\text{dist}}{k_{\text{slow}} \cdot r}$
}

$\mathbf{v}^{\text{pref}} \leftarrow v_{\text{limit}} \cdot \hat{\mathbf{d}}$
\end{algorithm}

\section*{Time-to-Collision (TTC) Calculation}
\begin{align}
\text{Relative position: } \mathbf{p}_{\text{rel}} &= \mathbf{p}_j - \mathbf{p}_i \\
\text{Relative velocity: } \mathbf{v}_{\text{rel}} &= \mathbf{v}_j - \mathbf{v}_i \\
\text{Combined radius: } R &= r_i + r_j \\
\text{TTC quadratic: } \|\mathbf{p}_{\text{rel}} + \mathbf{v}_{\text{rel}} t\|^2 &= R^2
\end{align}

Expanding the quadratic equation:
\begin{align}
a &= \|\mathbf{v}_{\text{rel}}\|^2 \\
b &= 2(\mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}) \\
c &= \|\mathbf{p}_{\text{rel}}\|^2 - R^2 \\
\text{TTC} &= \frac{-b - \sqrt{b^2 - 4ac}}{2a}
\end{align}

If $b^2 - 4ac \leq 0$ or TTC $\leq 0$, then TTC $= +\infty$.

\section*{TTC-Based Preemptive Braking}
\begin{algorithm}[H]
\caption{TTC Preemptive Velocity Scaling}
\KwIn{Robot state, Human states, TTC brake threshold $\text{TTC}_{\text{brake}}$}
\KwOut{Scaled preferred velocity $\mathbf{v}^{\text{pref}}_{\text{scaled}}$}

$\text{TTC}_{\min} \leftarrow +\infty$

\For{each human $h$ in humans}{
    $\mathbf{p}_{\text{rel}} \leftarrow (h.p_x - \text{robot}.p_x, h.p_y - \text{robot}.p_y)$
    $\mathbf{v}_{\text{rel}} \leftarrow (h.v_x - \text{robot}.v_x, h.v_y - \text{robot}.v_y)$
    $R \leftarrow h.r + \text{robot}.r$
    
    $\text{TTC} \leftarrow \text{compute\_ttc}(\mathbf{p}_{\text{rel}}, \mathbf{v}_{\text{rel}}, R)$
    $\text{TTC}_{\min} \leftarrow \min(\text{TTC}_{\min}, \text{TTC})$
}

\If{$\text{TTC}_{\min} < \text{TTC}_{\text{brake}}$ AND $\text{TTC}_{\min}$ is finite}{
    $\rho \leftarrow \text{clip}\left(\frac{\text{TTC}_{\min}}{\text{TTC}_{\text{brake}}}, \rho_{\min}, 1.0\right)$
    $\mathbf{v}^{\text{pref}}_{\text{scaled}} \leftarrow \rho \cdot \mathbf{v}^{\text{pref}}$
}
\Else{
    $\mathbf{v}^{\text{pref}}_{\text{scaled}} \leftarrow \mathbf{v}^{\text{pref}}$
}
\end{algorithm}

\section*{Symmetry Breaking}
\begin{align}
\theta &\sim \text{Uniform}(0, 2\pi) \\
\mathbf{j} &= \epsilon_{\text{noise}} \cdot (\cos \theta, \sin \theta) \\
\mathbf{v}^{\text{pref}}_{\text{final}} &= \mathbf{v}^{\text{pref}}_{\text{scaled}} + \mathbf{j}
\end{align}

where $\epsilon_{\text{noise}} = 0.03$ provides small random perturbations to break symmetrical deadlocks.

\section*{ORCA Simulator Configuration}
\begin{algorithm}[H]
\caption{ORCA Simulation Setup}
\KwIn{Robot state, Human states, ORCA parameters}
\KwOut{Collision-free velocity}

\tcp{Initialize or rebuild simulator if needed}
\If{simulator needs rebuild}{
    Create new RVO2 simulator with parameters:
    \begin{itemize}
        \item Time step: $\Delta t = 0.25$ s
        \item Neighbor distance: $d_{\text{neighbor}} = 3.0$ m
        \item Max neighbors: $N_{\max} = 8$
        \item Time horizon: $\tau = 4.0$ s
        \item Obstacle horizon: $\tau_{\text{obs}} = 3.0$ s
    \end{itemize}
}

\tcp{Add robot agent (index 0)}
$r_{\text{robot}} \leftarrow r + s_{\text{safety}} + s_{\text{label}}$ \tcp{Inflated radius}
Add robot with position, velocity, radius $r_{\text{robot}}$, max speed

\tcp{Add human agents (indices 1 to N)}
\For{each human $h$}{
    $r_h \leftarrow h.r + s_{\text{safety}}$
    Add human with position, velocity, radius $r_h$, max speed $h.v_{\text{pref}}$
}

\tcp{Set preferred velocities}
Set robot preferred velocity using goal-directed + TTC-scaled + noise
\For{each human $h$}{
    \If{human\_pref\_mode == 'goal'}{
        Set $\mathbf{v}^{\text{pref}}_h$ toward human's goal
    }
    \ElseIf{human\_pref\_mode == 'current'}{
        Set $\mathbf{v}^{\text{pref}}_h = \mathbf{v}_h$ (current velocity)
    }
    \Else{
        Set $\mathbf{v}^{\text{pref}}_h = (0, 0)$
    }
}

\tcp{Execute one ORCA step}
simulator.doStep()
$\mathbf{v}_{\text{new}} \leftarrow$ simulator.getAgentVelocity(0)
\Return{ActionXY($v_{\text{new},x}, v_{\text{new},y}$)}
\end{algorithm}

\section*{ORCA Velocity Obstacles}
The ORCA algorithm constructs velocity obstacles and finds the collision-free velocity closest to the preferred velocity:

\begin{align}
\text{Velocity Obstacle } VO_{i|j}^{\tau} &= \{\mathbf{v} : \exists t \in (0, \tau], \|\mathbf{p}_i + t\mathbf{v} - \mathbf{p}_j - t\mathbf{v}_j\| < r_i + r_j\} \\
\text{Reciprocal Velocity Obstacle } RVO_{i|j}^{\tau} &= \{\mathbf{v} : 2\mathbf{v} - \mathbf{v}_i - \mathbf{v}_j \in VO_{i|j}^{\tau}\} \\
\text{ORCA Half-plane } ORCA_{i|j}^{\tau} &= \{\mathbf{v} : (\mathbf{v} - \mathbf{v}_i^A) \cdot \mathbf{n}_{ij} \geq 0\}
\end{align}

where $\mathbf{n}_{ij}$ is the outward normal of the ORCA half-plane and $\mathbf{v}_i^A$ is a point on the boundary.

\section*{Expert Demonstration Generation}
\begin{algorithm}[H]
\caption{ORCA Expert Action Generation}
\KwIn{Joint state (robot + humans)}
\KwOut{Expert action for behavioral cloning}

\tcp{Extract states}
$\text{robot\_state} \leftarrow \text{state.self\_state}$
$\text{human\_states} \leftarrow \text{state.human\_states}$

\tcp{Configure ORCA simulator}
Setup ORCA simulator with inflated robot radius for conservative labels

\tcp{Compute expert action}
$\mathbf{v}^{\text{pref}} \leftarrow$ goal-directed velocity with deceleration
$\mathbf{v}^{\text{scaled}} \leftarrow$ TTC-based preemptive braking
$\mathbf{v}^{\text{final}} \leftarrow \mathbf{v}^{\text{scaled}} +$ symmetry-breaking noise

Set preferred velocities for all agents
$\mathbf{a}_{\text{expert}} \leftarrow$ ORCA.doStep()
\Return{$\mathbf{a}_{\text{expert}}$}
\end{algorithm}

\section*{Key Design Features}
\begin{itemize}
  \item \textbf{Conservative Radius Inflation}: Robot radius increased by $s_{\text{safety}} + s_{\text{label}} = 0.15$ m for safer demonstrations
  \item \textbf{TTC Preemptive Braking}: Reduces velocity when collision risk detected within 2 seconds
  \item \textbf{Goal-Directed Deceleration}: Linear speed reduction when approaching goal within $k_{\text{slow}} \cdot r$ distance
  \item \textbf{Symmetry Breaking}: Random noise $\epsilon_{\text{noise}} = 0.03$ prevents deadlocks
  \item \textbf{Adaptive Human Modeling}: Humans use goal-directed preferred velocities by default
\end{itemize}

\section*{Configuration Integration}
ORCA parameters are integrated with other system components:
\begin{itemize}
  \item \texttt{env.config}: Defines ORCA parameters and simulation environment
  \item \texttt{train.config}: IL episodes use ORCA for expert demonstrations
  \item \texttt{policy\_factory}: ORCA registered as 'orca' policy for expert data collection
  \item Action compatibility: ORCA outputs ActionXY objects compatible with environment
\end{itemize}

\end{document}
```