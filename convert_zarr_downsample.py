import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from numcodecs import Blosc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/stage5_train.zarr")
    ap.add_argument("--dst", default="data/stage5_train_192x256.zarr")
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256,
                    help="frames per interpolation batch")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_path, dst_path = Path(args.src), Path(args.dst)
    if dst_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst_path} exists; pass --overwrite to replace it.")
        shutil.rmtree(dst_path)

    src = zarr.open(str(src_path), mode="r")
    dst = zarr.open(str(dst_path), mode="w")

    # root attrs (update the recorded image dims to the new resolution)
    dst.attrs.update(dict(src.attrs))
    dst.attrs["img_height"] = args.height
    dst.attrs["img_width"] = args.width

    src_data, src_meta = src["data"], src["meta"]
    dst_data = dst.create_group("data")
    dst_meta = dst.create_group("meta")

    # meta: copy verbatim
    for key, val in src_meta.items():
        dst_meta.array(key, val[:], chunks=val.chunks,
                       compressor=val.compressor, dtype=val.dtype)

    # small data arrays (action/state/category_id/object_id): copy directly
    for key in src_data.keys():
        if key == "img":
            continue
        val = src_data[key]
        dst_data.array(key, val[:], chunks=val.chunks,
                       compressor=val.compressor, dtype=val.dtype)
        print(f"copied data/{key} {val.shape}")

    # img: downsample in batches, store uint8 at target resolution
    img_src = src_data["img"]
    n = img_src.shape[0]
    img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    img_dst = dst_data.zeros(
        "img",
        shape=(n, args.height, args.width, 3),
        chunks=(img_src.chunks[0], args.height, args.width, 3),
        dtype="uint8",
        compressor=img_compressor,
    )

    device = torch.device(args.device)
    print(f"downsampling img {tuple(img_src.shape)} -> "
          f"({n}, {args.height}, {args.width}, 3) on {device}")
    for i in range(0, n, args.batch):
        j = min(i + args.batch, n)
        batch = img_src[i:j]  # (b, H0, W0, 3) uint8
        with torch.no_grad():
            t = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255.0
            t = F.interpolate(t, size=(args.height, args.width),
                              mode="bilinear", align_corners=False)
            t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
            out = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        img_dst[i:j] = out
        if (i // args.batch) % 20 == 0:
            print(f"  {j}/{n} frames", flush=True)

    print(f"done -> {dst_path}")
    print("verify:", dict(dst.attrs).get("img_height"), dict(dst.attrs).get("img_width"),
          "| img", tuple(img_dst.shape), img_dst.dtype)


if __name__ == "__main__":
    main()
