import time, sys
from huggingface_hub import snapshot_download
for attempt in range(200):
    try:
        snapshot_download('Voxel51/WLASL', repo_type='dataset', local_dir='data/wlasl/raw', max_workers=12,
                          allow_patterns=['data/*', 'metadata.json'])
        print('DONE', flush=True); break
    except Exception as e:
        print('retry', attempt, str(e)[:100], flush=True); time.sleep(60)
