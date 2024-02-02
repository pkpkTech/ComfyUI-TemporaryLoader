import requests
import threading
import io
from urllib.parse import unquote
from tqdm.auto import tqdm

import safetensors.torch
import torch

import comfy.utils
import comfy.checkpoint_pickle

def download_chunk(url, start_byte, end_byte, result_parts, total_size, pbar_web, pbar_cli):
    thr = total_size // 10
    size = 0
    cnt = 0
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        with io.BytesIO() as part_data:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    part_data.write(chunk)
                    chunk_size = len(chunk)
                    size+=chunk_size
                    pbar_cli.update(chunk_size)
                    if size >= thr:
                        size = 0
                        if cnt < 9:
                            cnt+=1
                            pbar_web.update(1)

            result_parts[start_byte] = part_data.getvalue()
        del part_data
        pbar_web.update(10 - cnt)

def download_file(url, num_threads=4):
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        content_disposition = response.headers.get("Content-Disposition", None)
    except requests.exceptions.RequestException as e:
        return None, e
    finally:
        response.close()

    chunk_size = total_size // num_threads

    pbar = comfy.utils.ProgressBar(num_threads * 10 + 1)
    pbar.update_absolute(0)

    file_name = "blank"
    if content_disposition:
        # Content-Dispositionヘッダーからファイル名を抽出する
        parts = content_disposition.split(";")
        for part in parts:
            if part.strip().startswith("filename="):
                file_name = unquote(part.strip().split("=")[1].strip('"'))
                break

    threads = []
    result_parts = {}
    with tqdm(total=total_size) as pbar_cli:
        for i in range(num_threads):
            start_byte = chunk_size * i
            end_byte = start_byte + chunk_size - 1 if i < num_threads - 1 else ""
            chunk_total_size = total_size - start_byte if i == num_threads-1 else chunk_size
            thread = threading.Thread(target=download_chunk, args=(url, start_byte, end_byte, result_parts, chunk_total_size, pbar, pbar_cli))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    pbar.update_absolute(num_threads * 10 + 1)

    # 全てのチャンクがダウンロードできているか確認
    if len(result_parts) < num_threads:
        return None, None

    # ダウンロードされた部分データを結合してバイナリデータとして返す
    sorted_parts = sorted(result_parts.items())
    result = b"".join(part for start_byte, part in sorted_parts)

    return result, file_name

# Bin to torch
def load_torch_bin(bin, is_safetensors, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if is_safetensors:
        sd = safetensors.torch.load(bin)
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        with io.BytesIO(bin) as ckpt:
            if safe_load:
                pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
            else:
                pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd
