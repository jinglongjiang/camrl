# Contracts System LaTeX Documentation

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, algorithm2e}
\begin{document}

\section*{Notation Reference - Unified Data Contracts}
\begin{itemize}
  \item $\mathbf{T} \in \mathbb{R}^{B \times T \times 6 \times 13}$: Unified token tensor format
  \item $\mathbf{S} \in \mathbb{R}^{34}$: Joint state vector (9 robot + 25 human features)
  \item $\text{Batch6T13}$: Standardized batch data structure
  \item $B$: Batch size dimension
  \item $T$: Temporal sequence length
  \item $N_H = 5$: Maximum number of humans
  \item $D_R = 9$: Robot feature dimension
  \item $D_H = 5$: Human feature dimension per agent
  \item $D_{\text{token}} = 13$: Token feature dimension per agent
  \item $\text{pad\_mask} \in \{0,1\}^{B \times T}$: Sequence validity mask
\end{itemize}

\section*{Core Data Structure}
\begin{algorithm}[H]
\caption{Batch6T13 Data Contract}

\textbf{@dataclass}
\begin{itemize}
    \item $\mathbf{S}$: torch.Tensor $\in \mathbb{R}^{B \times T \times 6 \times 13}$ - Current state sequence
    \item $\mathbf{S2}$: torch.Tensor $\in \mathbb{R}^{B \times T \times 6 \times 13}$ - Next state sequence
    \item $\mathbf{A\_idx}$: torch.Tensor $\in \mathbb{Z}^{B \times T}$ - Action index sequence (tail-frame valid)
    \item $\text{pad}$: torch.Tensor $\in \{0,1\}^{B \times T}$ - Validity mask (True=valid position)
    \item $\mathbf{R}$: torch.Tensor $\in \mathbb{R}^{B \times 1}$ - Episode rewards
    \item $\mathbf{Dn}$: torch.Tensor $\in \{0,1\}^{B \times 1}$ - Done flags
\end{itemize}
\end{algorithm}

\section*{State Format Conversion}
\begin{algorithm}[H]
\caption{Universal State Conversion: to\_btnd()}
\KwIn{Input tensor $\mathbf{x}$ (various formats), human\_num=5}
\KwOut{Standardized tensor $\mathbf{T} \in \mathbb{R}^{B \times T \times 6 \times 13}$}

\tcp{Format detection and conversion}
\If{$\mathbf{x}.shape == (B, 34)$}{
    \tcp{Flat joint state format}
    $\mathbf{T} \leftarrow \text{joint34\_to\_tokens}(\mathbf{x})$ \tcp{[B, 1, 6, 13]}
}
\ElseIf{$\mathbf{x}.shape == (B, T, 34)$}{
    \tcp{Sequence joint state format}
    \For{$t = 0$ to $T-1$}{
        $\mathbf{T}[:, t, :, :] \leftarrow \text{joint34\_to\_tokens}(\mathbf{x}[:, t, :])$
    }
}
\ElseIf{$\mathbf{x}.shape == (B, 6, 13)$}{
    $\mathbf{T} \leftarrow \mathbf{x}.unsqueeze(1)$ \tcp{Add time dimension}
}
\ElseIf{$\mathbf{x}.shape == (B, T, 6, 13)$}{
    $\mathbf{T} \leftarrow \mathbf{x}$ \tcp{Already in target format}
}
\Else{
    Raise RuntimeError("Unsupported state format")
}

\Return{$\mathbf{T}$}
\end{algorithm}

\section*{Token Structure Definition}
\subsection*{Agent Token Layout (13 dimensions)}
\begin{align}
\text{Robot token (agent 0): } &[p_x, p_y, g_x, g_y, v_x, v_y, r, v_{\text{pref}}, \theta, 0, 0, 0, 1] \\
\text{Human token (agent i): } &[p_x^{\text{rel}}, p_y^{\text{rel}}, 0, 0, v_x, v_y, r, 0, 0, 0, 0, 0, \text{mask}]
\end{align}

\subsection*{Relative Coordinate Transformation}
\begin{algorithm}[H]
\caption{Joint State to Tokens Conversion}
\KwIn{Joint state $\mathbf{s} \in \mathbb{R}^{34}$}
\KwOut{Token tensor $\mathbf{T} \in \mathbb{R}^{1 \times 6 \times 13}$}

$\mathbf{r} \leftarrow \mathbf{s}[:9]$ \tcp{Robot state}
$\mathbf{H} \leftarrow \text{reshape}(\mathbf{s}[9:], (5, 5))$ \tcp{Human states}

\tcp{Robot token construction}
$\mathbf{T}[0, 0, :9] \leftarrow \mathbf{r}$
$\mathbf{T}[0, 0, 9:12] \leftarrow [0, 0, 0]$ \tcp{Reserved features}
$\mathbf{T}[0, 0, 12] \leftarrow 1$ \tcp{Valid mask}

\tcp{Human tokens with relative coordinates}
\For{$i = 1$ to $5$}{
    \If{$\|\mathbf{H}[i-1, :2]\|_2 > 0$}{
        $\mathbf{T}[0, i, 0:2] \leftarrow \mathbf{H}[i-1, :2] - \mathbf{r}[:2]$ \tcp{Relative position}
        $\mathbf{T}[0, i, 2:4] \leftarrow [0, 0]$ \tcp{No goal for humans}
        $\mathbf{T}[0, i, 4:7] \leftarrow \mathbf{H}[i-1, 2:5]$ \tcp{Velocity and radius}
        $\mathbf{T}[0, i, 7:12] \leftarrow [0, 0, 0, 0, 0]$ \tcp{Reserved features}
        $\mathbf{T}[0, i, 12] \leftarrow 1$ \tcp{Valid mask}
    }
    \Else{
        $\mathbf{T}[0, i, :] \leftarrow \mathbf{0}$ \tcp{Zero padding for absent humans}
    }
}
\end{algorithm}

\section*{Sequence Processing Utilities}
\subsection*{Batch Dimension Management}
\begin{align}
\text{ensure\_bt}(\mathbf{x}) &: \mathbb{R}^{*} \rightarrow \mathbb{R}^{B \times T} \\
\text{ensure\_bt1}(\mathbf{x}) &: \mathbb{R}^{*} \rightarrow \mathbb{R}^{B \times 1} \\
\text{build\_pad\_mask}(\mathbf{x}) &: \mathbb{R}^{B \times T \times *} \rightarrow \{0,1\}^{B \times T}
\end{align}

\begin{algorithm}[H]
\caption{Sequence Dimension Standardization}
\KwIn{Tensor $\mathbf{x}$ with arbitrary shape}
\KwOut{Standardized tensor with target shape}

\tcp{ensure\_bt: Force [B, T] shape}
\If{$\mathbf{x}$.dim() == 1}{
    $\mathbf{x} \leftarrow \mathbf{x}$.unsqueeze(0) \tcp{Add batch dimension}
}
\If{$\mathbf{x}$.dim() == 2 AND $\mathbf{x}$.size(1) != expected\_T}{
    $\mathbf{x} \leftarrow \mathbf{x}$.unsqueeze(1).expand(-1, expected\_T) \tcp{Broadcast time}
}

\tcp{ensure\_bt1: Extract tail frame for episode rewards}
\If{$\mathbf{x}$.dim() == 2 AND $\mathbf{x}$.size(1) > 1}{
    $\mathbf{x} \leftarrow \mathbf{x}[:, -1:]$ \tcp{Take last timestep}
}
\If{$\mathbf{x}$.dim() == 1}{
    $\mathbf{x} \leftarrow \mathbf{x}$.unsqueeze(1) \tcp{Add time dimension}
}
\end{algorithm}

\section*{Action Discretization Interface}
\begin{algorithm}[H]
\caption{Physical to Discrete Action Conversion}
\KwIn{Physical action $\mathbf{a}_{\text{phys}} \in \mathbb{R}^2$, Action space $\mathbf{A}_{\text{space}}$}
\KwOut{Discrete action index $a_{\text{idx}} \in \mathbb{Z}$}

\tcp{Find closest action in discrete space}
$\text{distances} \leftarrow [\|\mathbf{a}_{\text{phys}} - \mathbf{A}_{\text{space}}[i]\|_2 \text{ for } i \in \{0, ..., N-1\}]$

$a_{\text{idx}} \leftarrow \arg\min(\text{distances})$

\tcp{Validation check}
\If{$\min(\text{distances}) > \text{tolerance}$}{
    Log warning: "Action discretization error exceeds tolerance"
}

\Return{$a_{\text{idx}}$}
\end{algorithm}

\section*{Memory Buffer Integration}
\begin{algorithm}[H]
\caption{Sequence Replay Buffer Interface}
\KwIn{Episode trajectory, Buffer $\mathcal{D}_{\text{seq}}$}
\KwOut{Updated buffer with sequence data}

\tcp{Episode processing}
$\text{states\_seq} \leftarrow [\mathbf{s}_0, \mathbf{s}_1, ..., \mathbf{s}_{T-1}]$
$\text{actions\_seq} \leftarrow [a_0, a_1, ..., a_{T-2}]$ \tcp{T-1 actions}
$\text{reward} \leftarrow R_{\text{terminal}}$ \tcp{Episode-level reward}

\tcp{Convert to contract format}
$\mathbf{S} \leftarrow \text{to\_btnd}(\text{states\_seq})$ \tcp{[1, T, 6, 13]}
$\mathbf{S2} \leftarrow \text{to\_btnd}(\text{states\_seq}[1:])$ \tcp{Next states}
$\mathbf{A\_idx} \leftarrow \text{action\_to\_discrete\_index}(\text{actions\_seq})$
$\text{pad\_mask} \leftarrow \text{build\_pad\_mask}(\mathbf{S})$

\tcp{Store in buffer}
$\mathcal{D}_{\text{seq}}.\text{push}(\text{Batch6T13}(\mathbf{S}, \mathbf{S2}, \mathbf{A\_idx}, \text{pad\_mask}, R, D))$
\end{algorithm}

\section*{Interface Consistency Guarantees}
\subsection*{Shape Validation}
\begin{align}
\text{validate\_btnd}(\mathbf{T}) &: \text{Assert } \mathbf{T}.shape == (B, T, 6, 13) \\
\text{validate\_bt}(\mathbf{x}) &: \text{Assert } \mathbf{x}.shape == (B, T) \\
\text{validate\_bt1}(\mathbf{x}) &: \text{Assert } \mathbf{x}.shape == (B, 1)
\end{align}

\subsection*{Device Consistency}
\begin{algorithm}[H]
\caption{Device Alignment}
\KwIn{Tensors with mixed devices}
\KwOut{Device-consistent tensors}

$\text{target\_device} \leftarrow \text{policy.device}$

\For{each tensor $\mathbf{x}$ in batch}{
    \If{$\mathbf{x}$.device != target\_device}{
        $\mathbf{x} \leftarrow \mathbf{x}$.to(target\_device, non\_blocking=True)
    }
}
\end{algorithm}

\section*{Training-Inference Path Unification}
\begin{algorithm}[H]
\caption{Single Source of Truth Pattern}
\KwIn{Raw data (various formats)}
\KwOut{Consistent internal representation}

\tcp{Training path: Batch sequences}
\If{training\_mode}{
    $\text{batch} \leftarrow \text{memory.sample}()$
    $\text{batch\_standardized} \leftarrow \text{contracts.unpack\_batch}(\text{batch})$
    Process using standardized Batch6T13 format
}

\tcp{Inference path: Single observations}
\If{inference\_mode}{
    $\mathbf{T} \leftarrow \text{contracts.to\_btnd}(\text{observation})$
    $\mathbf{features} \leftarrow \text{encode}(\mathbf{T})$
    $\text{action} \leftarrow \text{select\_action}(\mathbf{features})$
}

\tcp{Evaluation path: Action-conditional}
\If{evaluation\_mode}{
    $\text{state\_seq}, \text{action\_seq}, \text{pad\_mask} \leftarrow \text{input}$
    $\mathbf{T} \leftarrow \text{contracts.to\_btnd}(\text{state\_seq})$
    $\mathbf{values} \leftarrow V(\mathbf{T}, \text{action\_seq})$
    \Return{$\mathbf{values}$}
}
\end{algorithm}

\section*{Error Prevention Mechanisms}
\subsection*{Format Validation}
\begin{align}
\text{Shape errors prevented: } &\text{RuntimeError: shape '[72, 1]' is invalid...} \\
\text{Stride errors prevented: } &\text{RuntimeError: view size not compatible...} \\
\text{Type errors prevented: } &\text{ERROR: 不支持的输入类型：tuple} \\
\text{Dimension errors prevented: } &\text{RuntimeError: [to\_btnd] 不支持的状态格式...}
\end{align}

\subsection*{Automatic Fallback Handling}
\begin{algorithm}[H]
\caption{Robust Input Processing}
\KwIn{Potentially malformed input}
\KwOut{Valid tensor or error}

\Try{
    $\mathbf{T} \leftarrow \text{primary\_conversion}(\text{input})$
}
\Catch{Exception $e$}{
    \tcp{Fallback with zero padding}
    Log warning: "Input format issue, using fallback: " + str(e)
    $\mathbf{T} \leftarrow \text{zero\_tensor}(B, T, 6, 13, \text{device}=\text{target\_device})$
}

\tcp{Final validation}
Assert $\mathbf{T}$.shape == (B, T, 6, 13)
Assert $\mathbf{T}$.device == target\_device
\Return{$\mathbf{T}$}
\end{algorithm}

\section*{Integration Points}
\subsection*{Training Pipeline Integration}
\begin{itemize}
  \item \textbf{train.py}: Uses contracts for batch unpacking and tensor standardization
  \item \textbf{memory.py}: Stores sequences in Batch6T13 format
  \item \textbf{mamba\_rl.py}: Handles dual-path processing with contracts
  \item \textbf{explorer.py}: Converts episodes to contract format before storage
\end{itemize}

\subsection*{Policy Integration}
\begin{align}
\text{Training forward: } &V((\mathbf{S}, \mathbf{A\_idx}, \text{pad\_mask})) \\
\text{Inference forward: } &\text{action} \leftarrow \text{policy.predict}(\text{joint\_state}) \\
\text{Evaluation forward: } &V(\text{batch\_data}) \text{ using contracts}
\end{align}

\section*{Performance Benefits}
\begin{itemize}
  \item \textbf{Eliminates Interface Inconsistencies}: Single conversion function prevents format mismatches
  \item \textbf{Reduces Debugging Time}: Standardized error messages and shape validation
  \item \textbf{Simplifies Development}: Unified contracts eliminate "拆东墙补西墙" issues
  \item \textbf{Improves Maintainability}: Clear data flow through standardized interfaces
  \item \textbf{Enables Systematic Fixes}: Contract-based approach prevents regression issues
\end{itemize}

\section*{Contracts API Reference}
\begin{align}
\text{to\_btnd}(x, \text{human\_num}) &: \text{Any format} \rightarrow \mathbb{R}^{B \times T \times 6 \times 13} \\
\text{ensure\_bt}(x) &: \text{Any format} \rightarrow \mathbb{R}^{B \times T} \\
\text{ensure\_bt1}(x) &: \text{Any format} \rightarrow \mathbb{R}^{B \times 1} \\
\text{build\_pad\_mask}(x) &: \mathbb{R}^{B \times T \times *} \rightarrow \{0,1\}^{B \times T} \\
\text{joint34\_to\_tokens}(\mathbf{s}) &: \mathbb{R}^{34} \rightarrow \mathbb{R}^{1 \times 6 \times 13} \\
\text{action\_to\_discrete\_index}(\mathbf{a}) &: \mathbb{R}^2 \rightarrow \mathbb{Z}
\end{align}

The contracts system provides a "Single Source of Truth" for data format handling, eliminating the interface inconsistencies that previously caused training instability and debugging difficulties.

\end{document}
```