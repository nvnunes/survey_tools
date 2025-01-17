#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import os
import time
import requests
import numpy as np

def download(url, filename, chunk_size=1024*1024, timeout=30, silent=False):
    try:
        response = requests.head(url, timeout=timeout)
        if response.status_code == 200:
            if not silent:
                print(f"Downloading: {url}")

            headers={}
            if os.path.isfile(f"{filename}.part"):
                current_size = os.path.getsize(f"{filename}.part")
                headers['Range'] = f'bytes={current_size}-'
            else:
                current_size = 0

            response = requests.get(url, headers=headers, stream=True, timeout=timeout)

            if not silent:
                num_chunks = 0
                start_chunks = np.ceil(current_size / chunk_size)
                total_size = current_size + int(response.headers.get('content-length', 0))
                total_chunks = np.ceil(total_size / chunk_size)
                start_time = time.time()

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(f"{filename}.part", 'ab') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    if not silent:
                        current_size += len(chunk)
                        num_chunks += 1
                        last_time = time.time()
                        avg_chunk_time = (last_time - start_time) / num_chunks
                        remaining_time = avg_chunk_time * (total_chunks - start_chunks - num_chunks)
                        print(f"\r {current_size}/{total_size} bytes ({current_size/total_size:.1%}, {remaining_time:.0f}s remaining)", end='', flush=True)

            if not silent:
                print(' complete', flush=True)

            os.rename(f"{filename}.part", filename)
            return True

    except requests.RequestException:
        pass

    return False
