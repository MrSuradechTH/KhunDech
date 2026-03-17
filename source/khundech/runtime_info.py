from pathlib import Path


def is_running_in_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    cgroup_path = Path("/proc/1/cgroup")
    if not cgroup_path.exists():
        return False
    content = cgroup_path.read_text(encoding="utf-8", errors="ignore")
    keywords = ("docker", "containerd", "kubepods")
    return any(key in content for key in keywords)


def get_runtime_facts() -> dict:
    return {
        "in_docker": is_running_in_docker(),
        "workdir": str(Path("/app")),
        "docker_socket": Path("/var/run/docker.sock").exists(),
        "skills_manifest": Path("/app/skills_manifest.json").exists(),
    }


def format_runtime_facts(facts: dict) -> str:
    lines = ["🧩 Runtime facts"]
    lines.append(f"- Running in Docker: {facts['in_docker']}")
    lines.append(f"- Working directory: {facts['workdir']}")
    lines.append(f"- Docker socket mounted: {facts['docker_socket']}")
    lines.append(f"- skills_manifest.json present: {facts['skills_manifest']}")
    return "\n".join(lines)