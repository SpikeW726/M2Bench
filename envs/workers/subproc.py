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
                if hasattr(env, "seed"):
                    child_conn.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    child_conn.send(None)
            elif cmd == "getattr":
                child_conn.send(getattr(env, data, None))
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
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
    
    def get_env_attr(self, key: str) -> Any:
        self.parent_conn.send(("getattr", key))
        return self.parent_conn.recv()
    
    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_conn.send(("setattr", {"key": key, "value": value}))
        self.parent_conn.recv()
    
    def seed(self, seed: int | None = None) -> list[int] | None:
        super().seed(seed)
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
