"""Download official OpenCV Zoo models; validate against Git LFS SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
FILES = {
    "yunet.onnx": "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
ANTI_SPOOFING_FILES = {
    "MiniFASNetV2.onnx": {
        "url": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
        "sha256": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
        "bytes": 1743581,
    },
    "MiniFASNetV1SE.onnx": {
        "url": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
        "sha256": "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
        "bytes": 1742335,
    },
}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "facial-prototype"}), timeout=90)


def main():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    with fetch("https://api.github.com/repos/opencv/opencv_zoo/commits/main") as response:
        revision = json.load(response)["sha"]
    manifest = {"repository": "opencv/opencv_zoo", "revision": revision, "files": {}}
    for name, remote in FILES.items():
        base = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{revision}/models/"
        with fetch(base + remote) as response:
            pointer = response.read().decode()
        digest = next(line.split("sha256:")[1] for line in pointer.splitlines() if line.startswith("oid "))
        size = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith("size ")))
        target = folder / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            print(f"Descargando {name} ({size / 1e6:.1f} MB)...", flush=True)
            temporary = target.with_suffix(".part")
            with fetch(f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{revision}/models/{remote}") as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Descarga incompleta o hash incorrecto: {name}")
            temporary.replace(target)
        with fetch(base + remote.split("/")[0] + "/LICENSE") as response:
            (folder / f"{name}.LICENSE").write_bytes(response.read())
        manifest["files"][name] = {"sha256": digest, "bytes": size, "source": remote}
        print(f"{name}: SHA256 verificado", flush=True)
    manifest["anti_spoofing"] = {
        "repository": "yakhyo/face-anti-spoofing",
        "license": "Apache-2.0",
        "files": {},
    }
    for name, metadata in ANTI_SPOOFING_FILES.items():
        target = folder / name
        digest = metadata["sha256"]
        size = metadata["bytes"]
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            print(f"Descargando {name} ({size / 1e6:.1f} MB)...", flush=True)
            temporary = target.with_suffix(".part")
            with fetch(metadata["url"]) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Descarga incompleta o hash incorrecto: {name}")
            temporary.replace(target)
        manifest["anti_spoofing"]["files"][name] = metadata
        print(f"{name}: SHA256 verificado", flush=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
