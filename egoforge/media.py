"""Local decoding and sampled visual QA; no visual semantic labels inferred."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .data import EDGES

CAMERAS = ["main_stereo_left", "main_stereo_right", "aux_left", "aux_right"]
COLORS = [(37, 217, 164), (242, 114, 165)]


def video_path(raw, episode_id, camera):
    return (
        Path(raw)
        / "videos/chunk-000"
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4"
    )


def overlay(frame, ep, i):
    frame = frame.copy()
    for hand in range(2):
        if not ep.present[i, hand]:
            continue
        xy = ep.keypoints[i, hand].round().astype(int)
        color = COLORS[hand]
        for a, b in EDGES:
            cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 3, cv2.LINE_AA)
        for x, y in xy:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return frame


def inspect_media(raw, out, ep, features, config):
    target = Path(out) / "media"
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "episode_id": ep.episode_id,
        "scope": "all decoded frames; image metrics sampled at one-second intervals",
        "cameras": {},
        "sampled_frames": [],
    }
    thumbs = []
    frame_hashes = {}
    for camera in CAMERAS:
        path = video_path(raw, ep.episode_id, camera)
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        metadata_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        samples = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            i = decoded
            decoded += 1
            if i % int(config.get("media_sample_stride", 30)):
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (384, 240))
            mean = float(gray.mean())
            blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
            clipped = float(np.mean((gray < 4) | (gray > 251)))
            tiny = cv2.resize(gray, (9, 8))
            bits = tiny[:, 1:] > tiny[:, :-1]
            h = "".join(f"{x:02x}" for x in np.packbits(bits))
            item = {
                "frame_index": i,
                "timestamp_s": i / fps,
                "camera": camera,
                "mean_luminance": mean,
                "laplacian_variance_384x240": blur,
                "clipped_fraction": clipped,
                "dhash": h,
                "flags": [],
            }
            if blur < 15:
                item["flags"].append("low_image_detail_candidate")
            if mean < 30:
                item["flags"].append("dark_frame_candidate")
            if clipped > 0.2:
                item["flags"].append("clipped_exposure_candidate")
            key = (camera, h)
            if key in frame_hashes:
                item["same_dhash_as_frame"] = frame_hashes[key]
            else:
                frame_hashes[key] = i
            samples.append(item)
            if camera == "main_stereo_left" and i in [0, 450, 900, 1350, 1800, 2400]:
                decorated = overlay(im, ep, i)
                cv2.imwrite(
                    str(target / f"frame_{i:06d}.jpg"),
                    decorated,
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )
                thumbs.append((i, cv2.cvtColor(decorated, cv2.COLOR_BGR2RGB)))
        cap.release()
        report["cameras"][camera] = {
            "metadata_frame_count": metadata_count,
            "decoded_frames": decoded,
            "fps": fps,
            "matches_pose_frame_count": decoded == len(ep.time),
            "sampled_frames": len(samples),
            "low_detail_count": sum(
                "low_image_detail_candidate" in x["flags"] for x in samples
            ),
            "clipped_exposure_count": sum(
                "clipped_exposure_candidate" in x["flags"] for x in samples
            ),
        }
        report["sampled_frames"].extend(samples)
    sheet = Image.new("RGB", (1600, 3 * 560 + 150), (12, 19, 29))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        title = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36
        )
    except OSError:
        font = title = ImageFont.load_default()
    draw.text(
        (30, 26), "EGOFORGE / Bimanual motion annotation", fill="#e9f1fa", font=title
    )
    draw.text(
        (30, 79),
        "Source: Humanola (CC BY 4.0) • inherited 2D skeletons + new machine phase proposals",
        fill="#a5b7cb",
        font=font,
    )
    for k, (i, im) in enumerate(thumbs):
        x = 25 + (k % 2) * 800
        y = 150 + (k // 2) * 560
        tile = Image.fromarray(im).resize((750, 469))
        sheet.paste(tile, (x, y))
        row = features.iloc[i]
        draw.text(
            (x, y + 480),
            f"{i / 30:05.1f}s  /  {row.get('state', 'proposal')}  /  quality {row.quality_score:.2f}",
            fill="#68e4c5",
            font=font,
        )
        text = " · ".join(row.qa_flags)[:75] or "No automated QA flags"
        draw.text((x, y + 516), text, fill="#b5c2d1", font=font)
    sheet.save(target / "contact_sheet.jpg", quality=91)
    return report
