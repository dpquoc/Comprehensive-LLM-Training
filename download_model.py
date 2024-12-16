from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='models', revision='master')