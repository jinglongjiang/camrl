# -*- coding: utf-8 -*-
"""
 - 

:
1. 
2. 
3. 
4. 
"""

import logging
from typing import Dict, Any, Optional, Union

class LogFormatter:
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self._format_count = 0
    
    def format_training_progress(self, episode: int, metrics: Dict[str, Any], 
                                phase: str = "TRAIN") -> str:
        """
        
        train.py, explorer.py
        
        Args:
            episode: episode
            metrics: 
            phase: 
        
        Returns:
            str: 
        """
        self._format_count += 1
        
        success = metrics.get('success_rate', metrics.get('succ', 0.0))
        collision = metrics.get('collision_rate', metrics.get('coll', 0.0))
        timeout = metrics.get('timeout_rate', metrics.get('timeout', 0.0))
        reward = metrics.get('total_reward', metrics.get('reward', 0.0))
        nav_time = metrics.get('nav_time', metrics.get('nav', 0.0))
        
        base_format = (f"{phase} ep={episode} | "
                      f"succ={success:.2f} coll={collision:.2f} timeout={timeout:.2f} "
                      f"nav={nav_time:.2f} reward={reward:+.4f}")
        
        if 'upd' in metrics or 'updates' in metrics:
            updates = metrics.get('upd', metrics.get('updates', 0))
            base_format += f" | upd={updates}"
        
        if 'eps' in metrics or 'epsilon' in metrics:
            eps = metrics.get('eps', metrics.get('epsilon', 0.0))
            base_format += f" eps={eps:.3f}"
        
        if 'gain' in metrics or 'action_gain' in metrics:
            gain = metrics.get('gain', metrics.get('action_gain', 0))
            base_format += f" gain={gain}"
        
        if 'buf' in metrics or 'buffer_size' in metrics:
            buf_size = metrics.get('buf', metrics.get('buffer_size', 0))
            base_format += f" [BUF] {buf_size}"
        
        if 'v_loss' in metrics:
            v_loss = metrics.get('v_loss', 0.0)
            base_format += f" | v_loss={v_loss:.4f}"
        
        if 'q_spread' in metrics:
            q_spread = metrics.get('q_spread', 0.0)
            base_format += f" q_spread={q_spread:.4f}"
        
        return base_format
    
    def format_evaluation_result(self, episode: int, eval_type: str, 
                                metrics: Dict[str, Any]) -> str:
        """
        
        explorer.py
        
        Args:
            episode: episode
            eval_type:  ('raw', 'safe', 'test', 'val')
            metrics: 
        
        Returns:
            str: 
        """
        success = metrics.get('success_rate', 0.0)
        collision = metrics.get('collision_rate', 0.0)
        timeout = metrics.get('timeout_rate', 0.0)
        nav_time = metrics.get('nav_time', 0.0)
        
        base_format = (f"EVAL[{eval_type}] ep={episode} | "
                      f"succ={success:.3f} coll={collision:.3f} timeout={timeout:.3f} "
                      f"nav={nav_time:.2f}")
        
        if 'safety_boost' in metrics:
            boost = metrics['safety_boost']
            base_format += f" (+{boost:.2f} safety boost)"
        
        return base_format
    
    def format_phase_summary(self, phase: str, metrics: Dict[str, Any],
                           extra_info: str = "") -> str:
        """
        debug.md step ③:  - 
        train.pyIL-EVAL

        Args:
            phase:  ('TRAIN', 'TEST', 'VAL')
            metrics: 
            extra_info: 

        Returns:
            str: （）
        """
        return ""
    
    def format_interaction_log(self, success: float, collision: float,
                              timeout: float, reward: float,
                              buffer_size: int, roll_stats: tuple = None) -> str:
        """
         ()
        train.pylog_train_interact

        Args:
            success, collision, timeout: 
            reward: 
            buffer_size: 
            roll_stats: (roll_success, roll_collision, roll_timeout) 

        Returns:
            str: 
        """
        line1 = (f"TRAIN(interact) | succ={success:.2f} coll={collision:.2f} "
                f"timeout={timeout:.2f} reward={reward:+.2f} | rl_buf={buffer_size}")

        if roll_stats:
            roll_s, roll_c, roll_t = roll_stats
            line2 = (f"TRAIN(rolling)  | succ={roll_s:.2f} coll={roll_c:.2f} "
                    f"timeout={roll_t:.2f} (50-episode curve)")
            return f"{line1}\n{line2}"
        else:
            return line1
    
    def format_config_snapshot(self, config: Dict[str, Any]) -> str:
        """
        
        train.pySOT
        
        Args:
            config: 
        
        Returns:
            str: 
        """
        lines = ["SOT SNAPSHOT:"]
        
        if all(k in config for k in ['epsilon_start', 'epsilon_end', 'epsilon_decay']):
            lines.append(f"  epsilon: {config['epsilon_start']:.3f}→"
                        f"{config['epsilon_end']:.3f} over {config['epsilon_decay']}ep")
        
        if all(k in config for k in ['train_batches', 'batch_size', 'rl_min_buf']):
            lines.append(f"  training: batches={config['train_batches']}, "
                        f"batch_size={config['batch_size']}, min_buf={config['rl_min_buf']}")
        
        if all(k in config for k in ['action_gain', 'stop_bias', 'T']):
            lines.append(f"  action: gain={config['action_gain']:.1f}, "
                        f"stop_bias={config['stop_bias']:.2f}, T={config['T']}")
        
        if all(k in config for k in ['success_reward', 'collision_penalty', 'w_prog_start', 'w_prog_end']):
            lines.append(f"  reward: success={config['success_reward']:.1f}, "
                        f"coll={config['collision_penalty']:.1f}, "
                        f"w_prog={config['w_prog_start']:.2f}→{config['w_prog_end']:.2f}")
        
        return '\n'.join(lines)
    
    def log_training_progress(self, episode: int, metrics: Dict[str, Any], 
                             phase: str = "TRAIN", level: int = logging.INFO):
        """
        
        
        Args:
            episode: episode
            metrics: 
            phase: 
            level: 
        """
        message = self.format_training_progress(episode, metrics, phase)
        self.logger.log(level, message)
    
    def log_evaluation_result(self, episode: int, eval_type: str, 
                             metrics: Dict[str, Any], level: int = logging.INFO):
        """
        
        """
        message = self.format_evaluation_result(episode, eval_type, metrics)
        self.logger.log(level, message)
    
    def log_phase_summary(self, phase: str, metrics: Dict[str, Any], 
                         extra_info: str = "", level: int = logging.INFO):
        """
        
        """
        message = self.format_phase_summary(phase, metrics, extra_info)
        self.logger.log(level, message)
    
    def log_interaction(self, success: float, collision: float, timeout: float,
                       reward: float, buffer_size: int, roll_stats: tuple = None,
                       level: int = logging.INFO):
        """
         ()
        """
        message = self.format_interaction_log(success, collision, timeout, reward,
                                              buffer_size, roll_stats)
        self.logger.log(level, message)
    
    def log_config_snapshot(self, config: Dict[str, Any], level: int = logging.INFO):
        """
        
        """
        message = self.format_config_snapshot(config)
        self.logger.log(level, message)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_formats': self._format_count,
            'logger_name': self.logger.name
        }


_global_log_formatter = None

def get_log_formatter(logger_name: str = "crowdnav") -> LogFormatter:
    global _global_log_formatter
    if _global_log_formatter is None:
        _global_log_formatter = LogFormatter(logger_name)
    return _global_log_formatter


def log_training_progress_unified(episode: int, metrics: Dict[str, Any], 
                                 phase: str = "TRAIN", logger_name: str = "crowdnav"):
    formatter = get_log_formatter(logger_name)
    formatter.log_training_progress(episode, metrics, phase)


def log_evaluation_result_unified(episode: int, eval_type: str, 
                                 metrics: Dict[str, Any], logger_name: str = "crowdnav"):
    formatter = get_log_formatter(logger_name)
    formatter.log_evaluation_result(episode, eval_type, metrics)


def log_phase_summary_unified(phase: str, metrics: Dict[str, Any], 
                             extra_info: str = "", logger_name: str = "crowdnav"):
    formatter = get_log_formatter(logger_name)
    formatter.log_phase_summary(phase, metrics, extra_info)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
    
    formatter = LogFormatter("test")
    
    train_metrics = {
        'succ': 0.75, 'coll': 0.20, 'timeout': 0.05,
        'reward': 5.23, 'nav': 12.5, 'upd': 16,
        'eps': 0.08, 'gain': 10, 'buf': 1024,
        'v_loss': 0.0234, 'q_spread': 0.067
    }
    formatter.log_training_progress(100, train_metrics)
    
    eval_metrics = {
        'success_rate': 0.85, 'collision_rate': 0.12, 
        'timeout_rate': 0.03, 'nav_time': 14.2,
        'safety_boost': 0.05
    }
    formatter.log_evaluation_result(100, "safe", eval_metrics)
    
    summary_metrics = {
        'success_rate': 0.82, 'collision_rate': 0.15,
        'timeout_rate': 0.03, 'nav_time': 13.8,
        'total_reward': 4.67, 'no_progress_rate': 0.0
    }
    formatter.log_phase_summary("TEST", summary_metrics, "in episode 100 ")
    
    config = {
        'epsilon_start': 0.15, 'epsilon_end': 0.08, 'epsilon_decay': 1500,
        'train_batches': 16, 'batch_size': 64, 'rl_min_buf': 512,
        'action_gain': 10.5, 'stop_bias': 0.20, 'T': 8,
        'success_reward': 20.0, 'collision_penalty': -8.0,
        'w_prog_start': 0.60, 'w_prog_end': 0.75
    }
    formatter.log_config_snapshot(config)
    
    stats = formatter.get_stats()
    print(f"\nFormatting stats: {stats}")
