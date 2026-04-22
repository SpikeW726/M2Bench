"""SubprocEnvWorker: 使用子进程运行环境，实现真正的并行。"""

import multiprocessing
import traceback
from multiprocessing import connection
from multiprocessing.context import BaseContext
from typing import Any, Callable, Literal, Tuple
import cloudpickle
import numpy as np
import gymnasium as gym

from envs.workers.base import EnvWorker

# 哨兵包装：子进程把捕获到的异常序列化后送回父进程，供父进程重新抛出
class _WorkerError:
    """子进程将异常封装为此对象后通过管道返回，父进程收到后重新抛出。"""
    def __init__(self, exc: BaseException, tb_str: str):
        self.exc = exc
        self.tb_str = tb_str


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
    """子进程工作函数，接收命令并执行环境操作。

    任何命令执行时抛出的异常均被捕获后以 _WorkerError 形式送回父进程，
    而不是让子进程直接崩溃（崩溃会导致父进程收到 EOFError 且看不到根因）。
    """
    parent_conn.close()
    env = env_fn_wrapper.data()

    def _safe_send(result):
        """发送结果；若 result 本身无法 pickle，则发送错误信息。"""
        try:
            child_conn.send(result)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                child_conn.send(_WorkerError(e, tb))
            except Exception:
                pass

    try:
        while True:
            try:
                cmd, data = child_conn.recv()
            except EOFError:
                break

            try:
                if cmd == "step":
                    _safe_send(env.step(data))
                elif cmd == "reset":
                    _safe_send(env.reset(**data))
                elif cmd == "close":
                    _safe_send(env.close())
                    break
                elif cmd == "render":
                    _safe_send(env.render(**data) if hasattr(env, "render") else None)
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
                        _safe_send(env.seed(sd))
                    else:
                        env.reset(seed=sd)
                        _safe_send(None)
                elif cmd == "getattr":
                    _safe_send(getattr(env, data, None))
                elif cmd == "setattr":
                    setattr(env.unwrapped, data["key"], data["value"])
                    _safe_send(None)
                elif cmd == "call_method":
                    # 在子进程中调用方法并返回结果（避免通过管道传递 bound method）
                    method_name, args, kwargs = data
                    method = getattr(env, method_name, None)
                    if method is not None and callable(method):
                        _safe_send(method(*args, **kwargs))
                    else:
                        _safe_send(None)
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
            except Exception as exc:
                # 命令执行出错：把异常序列化后送回父进程，子进程继续运行
                tb = traceback.format_exc()
                _safe_send(_WorkerError(exc, tb))
    except KeyboardInterrupt:
        pass
    finally:
        child_conn.close()


def _check_worker_result(result: Any) -> Any:
    """若结果是 _WorkerError，在父进程侧重新抛出，并附带子进程的完整 traceback。"""
    if isinstance(result, _WorkerError):
        raise RuntimeError(
            f"[SubprocWorker] Child process raised an exception:\n{result.tb_str}"
        ) from result.exc
    return result


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
        return _check_worker_result(self.parent_conn.recv())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.parent_conn.send(("step", action))
        return _check_worker_result(self.parent_conn.recv())
    
    # ---- 异步 step: 拆分为 send + recv，支持并行 ----
    def send_step(self, action: np.ndarray) -> None:
        """发送 step 命令，不等待结果"""
        self.parent_conn.send(("step", action))
    
    def recv_step(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """接收 step 结果；若子进程抛出异常则在此重新抛出（含完整 traceback）。"""
        return _check_worker_result(self.parent_conn.recv())
    
    def send_reset(self, **kwargs) -> None:
        """发送 reset 命令，不等待结果"""
        self.parent_conn.send(("reset", kwargs))
    
    def recv_reset(self) -> Tuple[np.ndarray, dict]:
        """接收 reset 结果；若子进程抛出异常则在此重新抛出（含完整 traceback）。"""
        return _check_worker_result(self.parent_conn.recv())
    
    def send_call_method(self, method_name: str, *args, **kwargs) -> None:
        """发送 call_method 命令，在子进程中调用方法"""
        self.parent_conn.send(("call_method", (method_name, args, kwargs)))
    
    def recv_call_method(self) -> Any:
        """接收 call_method 结果；若子进程抛出异常则在此重新抛出（含完整 traceback）。"""
        return _check_worker_result(self.parent_conn.recv())
    
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
            try:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
            except Exception:
                pass
            try:
                self.parent_conn.close()
            except Exception:
                pass
            try:
                self.process.close()
            except Exception:
                pass
