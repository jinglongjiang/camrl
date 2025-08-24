# Policy Framework LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Policy Framework}
\begin{itemize}
  \item $\pi$: Abstract policy interface
  \item $\mathcal{S}$: State space
  \item $\mathcal{A}$: Action space
  \item $\pi(\mathbf{a}|\mathbf{s})$: Policy probability distribution
  \item $\text{phase} \in \{\text{train}, \text{val}, \text{test}\}$: Training phase
  \item $\theta$: Policy parameters
  \item $\mathcal{D}$: Device (CPU/CUDA) for computation
  \item $\mathcal{E}$: Environment instance
\end{itemize}

\section*{Policy Base Class}
\begin{algorithm}[H]
\caption{Policy Interface}
\KwIn{State $\mathbf{s} \in \mathcal{S}$}
\KwOut{Action $\mathbf{a} \in \mathcal{A}$}

\tcp{Abstract methods that must be implemented}
\textbf{Abstract:} configure(config) \tcp{Initialize policy parameters}
\textbf{Abstract:} predict(state) $\rightarrow$ action \tcp{Main policy inference}

\tcp{Common interface methods}
set\_phase(phase) \tcp{Set training/evaluation mode}
set\_device(device) \tcp{Set computation device}
set\_env(env) \tcp{Set environment reference}
get\_model() $\rightarrow$ model \tcp{Return underlying model}

\tcp{Utility methods}
transform(state) $\rightarrow$ processed\_state \tcp{State preprocessing}
reach\_destination(state) $\rightarrow$ boolean \tcp{Goal reaching check}
\end{algorithm}

\section*{Policy Factory Registry}
The policy factory provides a unified interface for instantiating different navigation policies:

\begin{align}
\text{policy\_factory} = \{&\\
&\text{"cadrl"} \rightarrow \text{CADRL}, \\
&\text{"lstm\_rl"} \rightarrow \text{LstmRL}, \\
&\text{"sarl"} \rightarrow \text{SARL}, \\
&\text{"mamba"} \rightarrow \text{MambaRL}, \\
&\text{"mamba\_rl"} \rightarrow \text{MambaRL}, \text{ (alias)} \\
&\text{"orca"} \rightarrow \text{ORCA\_WRAPPER} \\
\}
\end{align}

\section*{Policy Instantiation}
\begin{algorithm}[H]
\caption{Policy Creation and Configuration}
\KwIn{Policy name, Configuration}
\KwOut{Configured policy instance}

\If{policy\_name not in policy\_factory}{
    Raise ValueError("Unknown policy: " + policy\_name)
}

$\text{PolicyClass} \leftarrow \text{policy\_factory}[\text{policy\_name}]$
$\pi \leftarrow \text{PolicyClass}(\text{config})$ \tcp{Instantiate}

\tcp{Standard configuration}
$\pi.$configure(config)
$\pi.$set\_device(device)
$\pi.$set\_phase(phase)

\If{environment available}{
    $\pi.$set\_env(env)
}

\Return{$\pi$}
\end{algorithm}

\section*{State Transformation}
\begin{algorithm}[H]
\caption{Generic State Transformation}
\KwIn{Raw state (various formats)}
\KwOut{Processed state for policy}

\If{state.has\_attr("to\_array")}{
    \tcp{JointState or similar structured state}
    $\text{state\_array} \leftarrow \text{state.to\_array}()$
}
\Else{
    \tcp{Already array/tensor format}
    $\text{state\_array} \leftarrow \text{state}$
}

\Return{state\_array}
\end{algorithm}

\section*{Goal Reaching Detection}
\begin{align}
\text{reach\_destination}(\mathbf{s}) &= \|\mathbf{p}_{\text{robot}} - \mathbf{g}_{\text{robot}}\|_2 < r_{\text{robot}} \\
\text{where: } \mathbf{p}_{\text{robot}} &= (s.\text{self\_state}.px, s.\text{self\_state}.py) \\
\mathbf{g}_{\text{robot}} &= (s.\text{self\_state}.gx, s.\text{self\_state}.gy) \\
r_{\text{robot}} &= s.\text{self\_state}.radius
\end{align}

\section*{Policy Lifecycle Management}
\begin{algorithm}[H]
\caption{Training Phase Management}

\tcp{Phase transitions during training}
\textbf{Initialize:} set\_phase("train")

\tcp{Expert data collection (IL phase)}
\While{collecting demonstrations}{
    set\_phase("train") with expert policy (ORCA)
    Use stochastic action sampling for exploration
}

\tcp{Reinforcement learning phase}  
\While{RL training}{
    set\_phase("train") with student policy
    Use exploration noise and entropy regularization
}

\tcp{Evaluation phases}
\For{validation episodes}{
    set\_phase("val")
    Use deterministic policy (no exploration)
}

\For{test episodes}{
    set\_phase("test")  
    Use deterministic policy (final evaluation)
}
\end{algorithm}

\section*{Device Management}
\begin{align}
\text{CPU mode: } &\mathcal{D} = \text{torch.device("cpu")} \\
\text{GPU mode: } &\mathcal{D} = \text{torch.device("cuda:0")} \\
\text{Transfer: } &\text{model.to}(\mathcal{D}), \text{data.to}(\mathcal{D})
\end{align}

For neural network policies, device management ensures:
\begin{itemize}
  \item Model parameters are on correct device
  \item Input tensors match model device  
  \item Consistent computation across training/inference
\end{itemize}

\section*{Policy Properties}
Each policy implementation defines key properties:

\begin{align}
\text{trainable} &\in \{\text{True}, \text{False}\} \text{ (whether policy learns)} \\
\text{multiagent\_training} &\in \{\text{True}, \text{False}\} \text{ (multi-human scenarios)} \\
\text{kinematics} &\in \{\text{"holonomic"}, \text{"differential"}\} \text{ (motion model)} \\
\text{action\_space} &\in \{\text{continuous}, \text{discrete}\} \text{ (action type)}
\end{align}

\section*{Policy-Specific Implementations}

\subsection*{CADRL (Collision Avoidance Deep RL)}
\begin{itemize}
  \item Multi-layer perceptron architecture
  \item Basic value-based navigation  
  \item Fixed human interaction modeling
\end{itemize}

\subsection*{LSTM-RL}
\begin{itemize}
  \item LSTM-based sequence modeling
  \item Temporal dependence in human behavior
  \item Recurrent state management
\end{itemize}

\subsection*{SARL (Socially Attentive RL)}
\begin{itemize}
  \item Attention mechanism for human interaction
  \item Social compliance modeling
  \item Multi-head attention architecture
\end{itemize}

\subsection*{MambaRL}
\begin{itemize}
  \item State-space model architecture (Mamba)
  \item Relative coordinate transformation
  \item Top-K human selection
  \item Advanced SAC-compatible design
\end{itemize}

\subsection*{ORCA (Expert Policy)}
\begin{itemize}
  \item Reciprocal velocity obstacles
  \item Classical collision avoidance
  \item Expert demonstrations for IL
  \item Non-trainable but configurable
\end{itemize}

\section*{Configuration Integration}
Policies read configuration from multiple sources:

\begin{align}
\text{env.config} &\rightarrow \text{Environment and reward parameters} \\
\text{policy.config} &\rightarrow \text{Network architecture and hyperparameters} \\
\text{train.config} &\rightarrow \text{Training procedures and schedules}
\end{align}

\section*{Policy Selection Logic}
\begin{algorithm}[H]
\caption{Training Pipeline Policy Selection}

\tcp{IL Phase: Expert data collection}
expert\_policy\_name $\leftarrow$ train.config["imitation\_learning"]["il\_policy"]
expert\_policy $\leftarrow$ policy\_factory[expert\_policy\_name](config)

\tcp{RL Phase: Student policy training}  
student\_policy\_name $\leftarrow$ args.policy
student\_policy $\leftarrow$ policy\_factory[student\_policy\_name](config)

\tcp{Evaluation: Use trained student policy}
eval\_policy $\leftarrow$ student\_policy
eval\_policy.set\_phase("test")
\end{algorithm}

\section*{Error Handling and Validation}
\begin{algorithm}[H]
\caption{Policy Validation}
\KwIn{Policy instance $\pi$}

\tcp{Required method validation}
Assert hasattr($\pi$, "configure")
Assert hasattr($\pi$, "predict")
Assert callable($\pi$.configure)
Assert callable($\pi$.predict)

\tcp{Training compatibility}
\If{training\_mode}{
    Assert $\pi$.trainable == True
}

\tcp{Device compatibility}
\If{using\_cuda}{
    Assert $\pi$.device.type == "cuda"
}
\end{algorithm}

\section*{Policy Interface Consistency}
All policies must implement consistent interfaces:

\begin{itemize}
  \item \texttt{predict(state) → action}: Core prediction method
  \item \texttt{configure(config)}: Parameter initialization
  \item \texttt{set\_device(device)}: Device management
  \item \texttt{set\_phase(phase)}: Training/evaluation mode switching
  \item \texttt{trainable}: Boolean flag for learning capability
  \item \texttt{multiagent\_training}: Multi-agent scenario support
\end{itemize}

This unified interface enables seamless policy switching during training and evaluation phases.

\end{document}
```