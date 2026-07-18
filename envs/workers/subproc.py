import multiprocessing
import traceback
from multiprocessing import connection
from multiprocessing.context import BaseContext
from typing import Any, Callable, Literal, Tuple
import cloudpickle
import numpy as np
import gymnasium as gym

from envs.workers.base import EnvWorker

class _WorkerError:
    def __init__(self, exc: BaseException, tb_str: str):
        self.exc = exc
        self.tb_str = tb_str

class CloudpickleWrapper:
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
    parent_conn.close()
    env = env_fn_wrapper.data()

    def _safe_send(result):
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

                    method_name, args, kwargs = data
                    method = getattr(env, method_name, None)
                    if method is not None and callable(method):
                        _safe_send(method(*args, **kwargs))
                    else:
                        _safe_send(None)
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
            except Exception as exc:

                tb = traceback.format_exc()
                _safe_send(_WorkerError(exc, tb))
    except KeyboardInterrupt:
        pass
    finally:
        child_conn.close()

def _check_worker_result(result: Any) -> Any:
    if isinstance(result, _WorkerError):
        raise RuntimeError(
            f"[SubprocWorker] Child process raised an exception:\n{result.tb_str}"
        ) from result.exc
    return result

class SubprocEnvWorker(EnvWorker):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        context: BaseContext | Literal["fork", "spawn"] | None = None,
    ):
        super().__init__(env_fn)

        if not isinstance(context, BaseContext):
            context = multiprocessing.get_context(context)

        self.parent_conn, child_conn = context.Pipe()

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

    def send_step(self, action: np.ndarray) -> None:
        self.parent_conn.send(("step", action))

    def recv_step(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return _check_worker_result(self.parent_conn.recv())

    def send_reset(self, **kwargs) -> None:
        self.parent_conn.send(("reset", kwargs))

    def recv_reset(self) -> Tuple[np.ndarray, dict]:
        return _check_worker_result(self.parent_conn.recv())

    def send_call_method(self, method_name: str, *args, **kwargs) -> None:
        self.parent_conn.send(("call_method", (method_name, args, kwargs)))

    def recv_call_method(self) -> Any:
        return _check_worker_result(self.parent_conn.recv())

    def get_env_attr(self, key: str) -> Any:
        self.parent_conn.send(("getattr", key))
        return self.parent_conn.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_conn.send(("setattr", {"key": key, "value": value}))
        self.parent_conn.recv()

    def seed(self, seed: int | None = None) -> list[int] | None:
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
