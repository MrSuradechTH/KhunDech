import json
import shlex
from datetime import datetime
from pathlib import Path

import docker
from docker.errors import APIError, DockerException, NotFound


def _truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _client() -> docker.DockerClient:
    if not Path("/var/run/docker.sock").exists():
        raise DockerException("Docker socket not mounted at /var/run/docker.sock")
    return docker.DockerClient(base_url="unix://var/run/docker.sock")


def _container_brief(container) -> str:
    name = container.name
    image = container.image.tags[0] if container.image.tags else container.image.short_id
    status = container.status
    return f"- {name} | {status} | {image}"


def _current_compose_project(client: docker.DockerClient) -> str | None:
    try:
        self_container = client.containers.get(Path("/etc/hostname").read_text(encoding="utf-8").strip())
    except Exception:
        return None
    labels = self_container.attrs.get("Config", {}).get("Labels", {})
    return labels.get("com.docker.compose.project")


def _self_container(client: docker.DockerClient):
    container_name = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    return client.containers.get(container_name)


def _compose_containers(client: docker.DockerClient):
    project = _current_compose_project(client)
    if not project:
        return []
    return client.containers.list(all=True, filters={"label": f"com.docker.compose.project={project}"})


def _run_docker_ps(client: docker.DockerClient, all_containers: bool) -> str:
    containers = client.containers.list(all=all_containers)
    if not containers:
        return "No containers found."
    lines = ["Docker containers:"]
    for container in containers:
        lines.append(_container_brief(container))
    return "\n".join(lines)


def _run_docker_logs(client: docker.DockerClient, name: str, tail: int = 120) -> str:
    container = client.containers.get(name)
    output = container.logs(tail=tail).decode("utf-8", errors="replace").strip()
    return output or "(no logs)"


def _run_docker_inspect(client: docker.DockerClient, name: str) -> str:
    container = client.containers.get(name)
    return json.dumps(container.attrs, indent=2, ensure_ascii=False)


def _run_container_lifecycle(client: docker.DockerClient, action: str, name: str) -> str:
    container = client.containers.get(name)
    if action == "start":
        container.start()
    elif action == "stop":
        container.stop()
    elif action == "restart":
        container.restart()
    else:
        raise ValueError(f"Unsupported lifecycle action: {action}")
    container.reload()
    return f"{container.name} -> {container.status}"


def _run_compose_ps(client: docker.DockerClient) -> str:
    containers = _compose_containers(client)
    if not containers:
        return "No compose containers found for current project."
    lines = ["Compose project containers:"]
    for container in containers:
        service = container.attrs.get("Config", {}).get("Labels", {}).get("com.docker.compose.service", "unknown")
        lines.append(f"- {service} | {container.name} | {container.status}")
    return "\n".join(lines)


def _run_compose_logs(client: docker.DockerClient, service_or_name: str | None, tail: int = 120) -> str:
    containers = _compose_containers(client)
    if not containers:
        return "No compose containers found for current project."

    selected = containers
    if service_or_name:
        service_or_name = service_or_name.lower()
        selected = []
        for container in containers:
            service = container.attrs.get("Config", {}).get("Labels", {}).get("com.docker.compose.service", "").lower()
            if container.name.lower() == service_or_name or service == service_or_name:
                selected.append(container)
    if not selected:
        return f"No compose container matched '{service_or_name}'."

    lines = []
    for container in selected:
        lines.append(f"### {container.name}")
        logs = container.logs(tail=tail).decode("utf-8", errors="replace").strip()
        lines.append(logs or "(no logs)")
    return "\n\n".join(lines)


def get_self_mounts() -> list[dict]:
    client = _client()
    client.ping()
    container = _self_container(client)
    mounts = container.attrs.get("Mounts", [])
    cleaned = []
    for mount in mounts:
        cleaned.append(
            {
                "type": mount.get("Type"),
                "source": mount.get("Source"),
                "destination": mount.get("Destination"),
                "mode": mount.get("Mode"),
                "rw": mount.get("RW"),
            }
        )
    return cleaned


def format_host_path_report() -> str:
    try:
        mounts = get_self_mounts()
    except DockerException as exc:
        return f"❌ Cannot read Docker mounts: {exc}"

    lines = ["🗂️ Host ↔ Container mount mapping"]
    app_mount = None
    for mount in mounts:
        lines.append(
            f"- {mount['destination']} <= {mount['source']} "
            f"(type={mount['type']}, rw={mount['rw']})"
        )
        if mount.get("destination") == "/app":
            app_mount = mount

    if app_mount:
        lines.append("")
        lines.append("✅ Your PC project path for this bot:")
        lines.append(f"- {app_mount['source']}")
    else:
        lines.append("")
        lines.append("⚠️ No /app bind mount found.")

    return "\n".join(lines)


def run_sync_probe() -> str:
    probe_path = Path("/app/data/sync_probe.txt")
    stamp = datetime.now().isoformat()
    value = f"sync-probe::{stamp}"
    probe_path.write_text(value, encoding="utf-8")

    mounts = get_self_mounts()
    app_source = None
    for mount in mounts:
        if mount.get("destination") == "/app":
            app_source = mount.get("source")
            break

    lines = ["🔁 Sync probe completed"]
    lines.append(f"- Container file: {probe_path}")
    if app_source:
        lines.append(f"- Host file: {app_source}\\data\\sync_probe.txt")
    lines.append(f"- Written value: {value}")
    lines.append("If you open that host file and see the same value, sync is active both ways.")
    return "\n".join(lines)


def restart_self() -> str:
    # Restart this bot's own Docker container via the Docker API.
    # Returns a status string; raises DockerException if the socket is unavailable.
    client = _client()
    client.ping()
    container = _self_container(client)
    name = container.name
    container.restart(timeout=5)  # gives the process 5 s to clean up
    return f"♻️ Docker restart issued to container: {name}"


def run_docker_command(args_text: str, use_compose: bool = False, timeout: int = 40) -> str:
    del timeout
    args = shlex.split(args_text) if args_text.strip() else []

    try:
        client = _client()
        client.ping()
    except DockerException as exc:
        return f"❌ Docker engine is unavailable: {exc}"

    try:
        if use_compose:
            subcmd = args[0] if args else "ps"
            if subcmd == "ps":
                return _truncate(_run_compose_ps(client))
            if subcmd == "logs":
                target = args[1] if len(args) >= 2 else None
                return _truncate(_run_compose_logs(client, target))
            return "❌ Supported !compose commands: ps, logs [service]"

        subcmd = args[0] if args else "ps"
        if subcmd == "ps":
            show_all = "-a" in args or "--all" in args
            return _truncate(_run_docker_ps(client, show_all))
        if subcmd == "logs" and len(args) >= 2:
            return _truncate(_run_docker_logs(client, args[1]))
        if subcmd == "inspect" and len(args) >= 2:
            return _truncate(_run_docker_inspect(client, args[1]))
        if subcmd in {"start", "stop", "restart"} and len(args) >= 2:
            return _truncate(_run_container_lifecycle(client, subcmd, args[1]))

        return (
            "❌ Supported !docker commands: "
            "ps [-a], logs <container>, inspect <container>, start <container>, stop <container>, restart <container>"
        )
    except (NotFound, APIError, DockerException, ValueError) as exc:
        return f"❌ Docker command failed: {exc}"