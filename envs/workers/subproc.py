"""SubprocEnvWorker: 使用子进程运行环境，实现真正的并行。"""

import multiprocessing
from multiprocessing import connection
from multiprocessing.context import BaseContext
from typing import Any, Callable, Literal, Tuple
import cloudpickle
import numpy as np
import gymnasium as gym

from envs.workers.base import EnvWorker


class CloudpickleWrapper:
    """用于序列化环境创建函数的包装器。"""
    
    def __init__(self, data: Any):
        self.data = data
    
    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)
    
    def __setstate__(self, data: bytes) -> None:
        self.data = cloudpickle.loads(data)


def _worker(
    parent_conn: connection.Connection,
    child_conn: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """子进程工作函数，接收命令并执行环境操作。"""
    parent_conn.close()
    env = env_fn_wrapper.data()
    
    try:
        while True:
            try:
                cmd, data = child_conn.recv()
            except EOFError:
                break
            
            if cmd == "step":
                child_conn.send(env.step(data))
            elif cmd == "reset":
                child_conn.send(env.reset(**data))
            elif cmd == "close":
                child_conn.send(env.close())
                break
            elif cmd == "render":
                child_conn.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                sd = data
                # 必须在子进程内 seed：action_space(agent) 经管道 pickle 会变成裸 function
                if getattr(env, "possible_agents", None):
                    for agent in env.possible_agents:
                        try:
                            sp = env.action_space(agent)
                        except Exception:
                            continue
                        if sp is not None and hasattr(sp, "seed"):
                            sp.seed(sd)
                else:
                    asp = getattr(env, "action_space", None)
                    if asp is not None and hasattr(asp, "seed"):
                        asp.seed(sd)
                if hasattr(env, "seed"):
                    child_conn.send(env.seed(sd))
                else:
                    env.reset(seed=sd)
                    child_conn.send(None)
            elif cmd == "getattr":
                child_conn.send(getattr(env, data, None))
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
                child_conn.send(None)
            elif cmd == "call_method":
                # 在子进程中调用方法并返回结果（避免通过管道传递 bound method）
                method_name, args, kwargs = data
                method = getattr(env, method_name, None)
                if method is not None and callable(method):
                    child_conn.send(method(*args, **kwargs))
                else:
                    child_conn.send(None)
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        child_conn.close()


class SubprocEnvWorker(EnvWorker):
    """使用子进程运行环境的 worker。"""
    
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        context: BaseContext | Literal["fork", "spawn"] | None = None,
    ):
        super().__init__(env_fn)
        
        # 获取 multiprocessing context
        if not isinstance(context, BaseContext):
            context = multiprocessing.get_context(context)
        
        # 创建管道
        self.parent_conn, child_conn = context.Pipe()
        
        # 启动子进程
        self.process = context.Process(
            target=_worker,
            args=(self.parent_conn, child_conn, CloudpickleWrapper(env_fn)),
            daemon=True,
        )
        self.process.start()
        child_conn.close()
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.parent_conn.send(("reset", kwargs))
        return self.parent_conn.recv()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.parent_conn.send(("step", action))
        return self.parent_conn.recv()
    
    # ---- 异步 step: 拆分为 send + recv，支持并行 ----
    def send_step(self, action: np.ndarray) -> None:
        """发送 step 命令，不等待结果"""
        self.parent_conn.send(("step", action))
    
    def recv_step(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """接收 step 结果"""
        return self.parent_conn.recv()
    
    def send_reset(self, **kwargs) -> None:
        """发送 reset 命令，不等待结果"""
        self.parent_conn.send(("reset", kwargs))
    
    def recv_reset(self) -> Tuple[np.ndarray, dict]:
        """接收 reset 结果"""
        return self.parent_conn.recv()
    
    def send_call_method(self, method_name: str, *args, **kwargs) -> None:
        """发送 call_method 命令，在子进程中调用方法"""
        self.parent_conn.send(("call_method", (method_name, args, kwargs)))
    
    def recv_call_method(self) -> Any:
        """接收 call_method 结果"""
        return self.parent_conn.recv()
    
    def get_env_attr(self, key: str) -> Any:
        self.parent_conn.send(("getattr", key))
        return self.parent_conn.recv()
    
    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_conn.send(("setattr", {"key": key, "value": value}))
        self.parent_conn.recv()
    
    def seed(self, seed: int | None = None) -> list[int] | None:
        # 不在父进程调 EnvWorker.seed：get_env_attr("action_space") 无法跨进程还原 bound method
        self.parent_conn.send(("seed", seed))
        return self.parent_conn.recv()
    
    def render(self, **kwargs) -> Any:
        self.parent_conn.send(("render", kwargs))
        return self.parent_conn.recv()
    
    def close_env(self) -> None:
        try:
            self.parent_conn.send(("close", None))
            self.parent_conn.recv()
            self.process.join(timeout=1)
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
