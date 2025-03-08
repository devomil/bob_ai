from core.model_downloader import ModelDownloader

# Set up downloader for GPT-J
gptj_downloader = ModelDownloader("EleutherAI/gpt-j-6B")

# Download and save GPT-J model
gptj_downloader.download_and_save(is_causal_lm=True)
