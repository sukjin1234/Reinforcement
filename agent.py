"""Read-only host metrics agent. Run one on each inference host."""
import os
import subprocess
import time
import psutil
from fastapi import FastAPI, HTTPException, Header

app = FastAPI()
psutil.cpu_percent(interval=None)


def gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                timeout=2, check=True)
        gpus = []
        for line in result.stdout.strip().splitlines():
            util, used, total = [float(x.strip()) for x in line.split(',')]
            if total <= 0:
                raise ValueError('invalid GPU memory')
            gpus.append({'utilization_pct': util, 'memory_used_mib': used, 'memory_total_mib': total})
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


@app.get('/metrics')
def metrics(authorization: str | None = Header(default=None)):
    token = os.environ.get('METRICS_TOKEN')
    if token and authorization != 'Bearer ' + token:
        raise HTTPException(401, 'Unauthorized')
    memory = psutil.virtual_memory()
    return {'timestamp': time.time(), 'cpu_pct': psutil.cpu_percent(interval=None),
            'memory_pct': memory.percent, 'memory_available_bytes': memory.available,
            'gpus': gpu_info()}
