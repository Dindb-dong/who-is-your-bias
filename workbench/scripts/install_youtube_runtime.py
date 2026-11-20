"""Install the official Deno binary inside this project, checking its published digest."""
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import requests

root = Path(__file__).resolve().parents[1]
platforms = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}


def get(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response


def main():
    key = (platform.system(), platform.machine())
    if key not in platforms:
        raise SystemExit("이 장치는 https://docs.deno.com/runtime/getting_started/installation/ 을 참고하세요.")
    metadata = get("https://api.github.com/repos/denoland/deno/releases/latest").json()
    filename = "deno-" + platforms[key] + ".zip"
    assets = {a["name"]: a["browser_download_url"] for a in metadata["assets"]}
    archive = get(assets[filename]).content
    checksum = get(assets[filename + ".sha256sum"]).text.split()[0]
    if hashlib.sha256(archive).hexdigest() != checksum:
        raise SystemExit("공식 체크섬과 일치하지 않아 설치하지 않았습니다.")
    executable = "deno.exe" if platform.system() == "Windows" else "deno"
    directory = root / ".tools"
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        (directory / executable).write_bytes(bundle.read(executable))
    (directory / executable).chmod(0o755)
    (directory / "deno-version.json").write_text(json.dumps({"version": metadata["tag_name"], "sha256": checksum, "source": assets[filename]}, indent=2))
    print("프로젝트 내부에 설치했습니다:", directory / executable)


if __name__ == "__main__":
    main()
