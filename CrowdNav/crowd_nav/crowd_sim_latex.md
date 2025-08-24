# CrowdSim Environment LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - CrowdSim Environment}
\begin{itemize}
  \item $R$: Robot agent with position $(p_x^R, p_y^R)$, goal $(g_x^R, g_y^R)$, velocity $(v_x^R, v_y^R)$
  \item $H_i$: Human agent $i$ with position $(p_x^i, p_y^i)$, velocity $(v_x^i, v_y^i)$, radius $r_i$
  \item $N_H = 5$: Maximum number of humans (from env.config)
  \item $T = 100$: Time limit in seconds (from env.config)
  \item $\Delta t = 0.25$: Time step duration (from env.config)
  \item $r_R$: Robot radius, $r_i$: Human $i$ radius
  \item $\mathbf{a} = (v_x, v_y)$: Robot action (velocity command)
  \item $d_{\min}$: Minimum separation distance between robot and humans
  \item $\text{TTC}_{\min}$: Minimum time-to-collision
  \item $\mathbf{s} \in \mathbb{R}^{34}$: Joint observation state
\end{itemize}

\section*{Reward Function Components}
\subsection*{Base Terminal Rewards}
From \texttt{env.config}:
\begin{align}
R_{\text{success}} &= 10.0 \\
R_{\text{collision}} &= -25.0 \\
R_{\text{timeout}} &= 0.0
\end{align}

\subsection*{Reward Weights}
From \texttt{env.config}:
\begin{align}
w_{\text{prog}} &= 0.8 & \text{(progress weight)} \\
w_{\text{goal}} &= 3.0 & \text{(goal weight)} \\
w_{\text{coll}} &= 12.0 & \text{(collision weight)} \\
w_{\text{soc}} &= 5.0 & \text{(social discomfort)} \\
w_{\text{time}} &= 0.01 & \text{(time cost)} \\
w_{\text{ttc}} &= 6.0 & \text{(time-to-collision)} \\
w_{\text{speed}} &= 0.25 & \text{(speed deviation)} \\
w_{\text{relv}} &= 2.5 & \text{(relative velocity)}
\end{align}

\section*{Progress Calculation}
\begin{align}
d_{\text{prev}} &= \|\mathbf{p}_R(t) - \mathbf{g}_R\|_2 \\
d_{\text{curr}} &= \|\mathbf{p}_R(t+\Delta t) - \mathbf{g}_R\|_2 \\
\text{progress} &= d_{\text{prev}} - d_{\text{curr}}
\end{align}

\section*{Collision Detection}
\begin{algorithm}[H]
\caption{Collision and Distance Computation}
\KwIn{Robot position $\mathbf{p}_R$, Human positions $\{\mathbf{p}_i\}$, Action $\mathbf{a}$}
\KwOut{Collision flag, minimum distance $d_{\min}$, closest human ID}

$d_{\min} \leftarrow +\infty$, $\text{collision} \leftarrow \text{False}$, $\text{closest\_id} \leftarrow -1$

\For{$i = 1$ to $N_H$}{
    $\mathbf{p}_{\text{rel}} \leftarrow \mathbf{p}_i - \mathbf{p}_R$ \tcp{Relative position}
    $\mathbf{v}_{\text{rel}} \leftarrow \mathbf{v}_i - \mathbf{a}$ \tcp{Relative velocity}
    $\mathbf{p}_{\text{next}} \leftarrow \mathbf{p}_{\text{rel}} + \mathbf{v}_{\text{rel}} \cdot \Delta t$ \tcp{Next relative pos}
    
    $d \leftarrow \text{point\_to\_segment\_dist}(\mathbf{p}_{\text{rel}}, \mathbf{p}_{\text{next}}, \mathbf{0}) - r_R - r_i$
    
    \If{$d < 0$}{
        $\text{collision} \leftarrow \text{True}$
        $d_{\min} \leftarrow d$, $\text{closest\_id} \leftarrow i$
        \textbf{break}
    }
    \If{$d < d_{\min}$}{
        $d_{\min} \leftarrow d$, $\text{closest\_id} \leftarrow i$
    }
}
\end{algorithm}

\section*{Time-to-Collision (TTC) Calculation}
\begin{align}
\text{TTC}(\mathbf{p}, \mathbf{v}, R) &= \frac{-b - \sqrt{b^2 - 4ac}}{2a} \\
\text{where: } a &= \|\mathbf{v}\|^2 \\
b &= 2(\mathbf{p} \cdot \mathbf{v}) \\
c &= \|\mathbf{p}\|^2 - R^2 \\
R &= r_R + r_i \text{ (combined radius)}
\end{align}

If $b^2 - 4ac \leq 0$ or result $\leq 0$, then $\text{TTC} = +\infty$.

\section*{Comprehensive Reward Function}
\begin{algorithm}[H]
\caption{Reward Calculation}
\KwIn{Current state, action $\mathbf{a}$, next state}
\KwOut{Total reward $r$}

$r \leftarrow 0$

\tcp{Terminal conditions}
\If{$t \geq T$}{
    $r \leftarrow r - w_{\text{goal}} \cdot 0.1$ \tcp{Timeout penalty}
    \Return{$r$, truncated}
}
\ElseIf{collision detected}{
    $r \leftarrow r + w_{\text{coll}} \cdot R_{\text{collision}}$
    \Return{$r$, terminated}
}
\ElseIf{goal reached}{
    $r \leftarrow r + w_{\text{goal}} \cdot R_{\text{success}}$
    \Return{$r$, terminated}
}
\Else{
    \tcp{Continuing episode - apply shaping rewards}
    
    \tcp{1. Progress reward}
    $r \leftarrow r + w_{\text{prog}} \cdot \text{progress}$
    
    \tcp{2. Time cost}
    $r \leftarrow r - w_{\text{time}} \cdot \Delta t$
    
    \tcp{3. Social discomfort (quadratic penalty)}
    \If{$d_{\min} < d_{\text{discomfort}}$}{
        $\text{penalty} \leftarrow \text{factor} \cdot (d_{\text{discomfort}} - d_{\min})^2$
        $r \leftarrow r - w_{\text{soc}} \cdot \text{penalty}$
    }
    
    \tcp{4. TTC penalty}
    \If{$w_{\text{ttc}} > 0$ AND $\text{TTC}_{\min} < \text{TTC}_{\text{thresh}}$}{
        $r \leftarrow r - w_{\text{ttc}} \cdot \frac{\text{TTC}_{\text{thresh}} - \text{TTC}_{\min}}{\text{TTC}_{\text{thresh}}}$
    }
    
    \tcp{5. Speed deviation penalty}
    \If{$w_{\text{speed}} > 0$}{
        $v_{\text{obs}} \leftarrow \|\mathbf{a}\|_2$
        $v_{\text{target}} \leftarrow \text{speed\_target} \cdot v_{\text{pref}}$
        $\text{deviation} \leftarrow \frac{v_{\text{obs}} - v_{\text{target}}}{v_{\text{pref}}}$
        $r \leftarrow r - w_{\text{speed}} \cdot \text{deviation}^2$
    }
    
    \tcp{6. Relative velocity penalty}
    \If{$w_{\text{relv}} > 0$ AND $d_{\min} < d_{\text{social}}$}{
        $v_{\text{rel}} \leftarrow \|\mathbf{v}_{\text{closest}} - \mathbf{a}\|_2$
        $r \leftarrow r - w_{\text{relv}} \cdot \max(0, v_{\text{rel}} - v_{\text{safe}})$
    }
}
\end{algorithm}

\section*{Action Conversion}
\begin{align}
\mathbf{a}_{\text{norm}} &\in [-1, 1]^2 \text{ (from neural network)} \\
\sigma_{\text{scale}} &= 0.65 \text{ (from policy.config)} \\
v_{\text{pref}} &= 1.0 \text{ (from env.config)} \\
\mathbf{a}_{\text{phys}} &= \mathbf{a}_{\text{norm}} \cdot v_{\text{pref}} \cdot \sigma_{\text{scale}} \\
\text{ActionXY}(v_x, v_y) &= \text{ActionXY}(a_{\text{phys}}[0], a_{\text{phys}}[1])
\end{align}

\section*{Observation Space Construction}
\begin{algorithm}[H]
\caption{Observation Assembly}
\KwOut{Joint observation $\mathbf{s} \in \mathbb{R}^{34}$}

$\mathbf{s}_R \leftarrow [p_x^R, p_y^R, g_x^R, g_y^R, v_x^R, v_y^R, r_R, v_{\text{pref}}^R, \theta_R]$ \tcp{Robot: 9D}

\For{$i = 1$ to $N_H$}{
    $\mathbf{s}_i \leftarrow [p_x^i, p_y^i, v_x^i, v_y^i, r_i]$ \tcp{Human $i$: 5D}
}

$\mathbf{s} \leftarrow [\mathbf{s}_R; \mathbf{s}_1; \mathbf{s}_2; \mathbf{s}_3; \mathbf{s}_4; \mathbf{s}_5]$ \tcp{Total: 34D}
\end{algorithm}

\section*{Position Generation}
\subsection*{Circle Crossing}
\begin{align}
\text{Robot: } &(0, -R_{\text{circle}}) \rightarrow (0, +R_{\text{circle}}) \\
\text{Human } i: &(R_{\text{circle}} \cos \theta_i, R_{\text{circle}} \sin \theta_i) \rightarrow (-R_{\text{circle}} \cos \theta_i, -R_{\text{circle}} \sin \theta_i) \\
\text{where: } &\theta_i \sim \text{Uniform}(0, 2\pi) \\
&R_{\text{circle}} = 4.0 \text{ (from env.config)}
\end{align}

\subsection*{Square Crossing}
\begin{align}
\text{Width} &= 10.0 \text{ (from env.config)} \\
\text{Human } i: &\text{Start: } (s \cdot W/2 \cdot \text{rand}, (\text{rand}-0.5) \cdot W) \\
&\text{Goal: } (-s \cdot W/2 \cdot \text{rand}, (\text{rand}-0.5) \cdot W) \\
\text{where: } &s \in \{-1, +1\} \text{ randomly}, \text{rand} \sim \text{Uniform}(0,1)
\end{align}

\section*{Configuration Parameters}
From \texttt{env.config}:
\begin{itemize}
  \item \texttt{time\_limit = 100}: Episode time limit $T$
  \item \texttt{time\_step = 0.25}: Simulation time step $\Delta t$
  \item \texttt{success\_reward = 10.0}: Success terminal reward $R_{\text{success}}$
  \item \texttt{collision\_penalty = -25.0}: Collision penalty $R_{\text{collision}}$
  \item \texttt{discomfort\_dist = 0.35}: Social discomfort threshold $d_{\text{discomfort}}$
  \item \texttt{discomfort\_penalty\_factor = 10.0}: Social penalty scaling factor
  \item \texttt{ttc\_thresh = 4.0}: TTC warning threshold $\text{TTC}_{\text{thresh}}$
  \item \texttt{speed\_target = 0.4}: Target speed ratio $\text{speed\_target}$
  \item \texttt{noprog\_eps = 0.015}: No-progress threshold
  \item \texttt{noprog\_patience = 24}: No-progress patience steps
  \item \texttt{circle\_radius = 4}: Circle crossing radius $R_{\text{circle}}$
  \item \texttt{square\_width = 10}: Square crossing width $W$
  \item \texttt{human\_num = 5}: Number of humans $N_H$
\end{itemize}

\section*{Episode Termination Conditions}
\begin{align}
\text{Success: } &\|\mathbf{p}_R - \mathbf{g}_R\|_2 < r_R \\
\text{Collision: } &\exists i: \|\mathbf{p}_R - \mathbf{p}_i\|_2 < r_R + r_i \\
\text{Timeout: } &t \geq T \\
\text{No Progress: } &\text{progress} < \epsilon_{\text{noprog}} \text{ for } n_{\text{patience}} \text{ consecutive steps}
\end{align}

where $\epsilon_{\text{noprog}} = 0.015$ and $n_{\text{patience}} = 24$ from config.

\end{document}
```