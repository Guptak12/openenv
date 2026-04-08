"""FastAPI app entrypoint for the CLI Auto Fixer environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to serve this environment. Install project dependencies first."
    ) from exc

try:
    from ..env import CliAutoFixerEnvironment
    from ..models import CliAutoFixerAction, CliAutoFixerObservation
except ImportError:
    from env import CliAutoFixerEnvironment
    from models import CliAutoFixerAction, CliAutoFixerObservation


app = create_app(
    CliAutoFixerEnvironment,
    CliAutoFixerAction,
    CliAutoFixerObservation,
    env_name="cli_auto_fixer",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
