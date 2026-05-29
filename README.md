<div align="center">
  <img src="docs/assets/blast_icon_only.png" width="200" height="200" alt="BLAST Logo">
</div>

<p align="center" style="font-size: 24px">A high-performance serving engine for web browsing AI.</p>

<div align="center">

[![Website](https://img.shields.io/badge/blastproject.org-FFE067)](https://blastproject.org)
[![Documentation](https://img.shields.io/badge/Docs-FFE067)](https://docs.blastproject.org)
[![Discord](https://img.shields.io/badge/Discord-FFE067)](https://discord.gg/NqrkJwYYh4)
[![Twitter Follow](https://img.shields.io/twitter/follow/realcalebwin?style=social)](https://x.com/realcalebwin)

</div>

<div align="center">
  <img src="website/assets/blast_ui_gif.gif" alt="BLAST UI Demo" width="80%">
</div>

## Use Cases

1. **I want to add web browsing AI to my app...** BLAST serves web browsing AI with an OpenAI-compatible API and concurrency and streaming baked in.
2. **I need to automate workflows...** BLAST will automatically cache and parallelize to keep costs down and enable interactive-level latencies.
3. **Just want to use this locally...** BLAST makes sure you stay under budget and not hog your computer's memory.

## Quick Start

```bash
pip install blastai && blastai serve
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://127.0.0.1:8000"
)

# Stream real-time browser actions
stream = client.responses.create(
    model="not-needed",
    input="Compare fried chicken reviews for top 10 fast food restaurants",
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta if " " in event.delta else "<screenshot>", end="", flush=True)
```

## Features

- **OpenAI-Compatible API** Drop-in replacement for OpenAI's API
- **High Performance** Automatic parallelism and prefix caching
- **Streaming** Stream browser-augmented LLM output to users
- **Concurrency** Out-of-the-box support many users with efficient resource management

## Documentation

Visit [documentation](https://docs.blastproject.org) to learn more.

## Contributing

Awesome! See our [Contributing Guide](https://docs.blastproject.org/development/contributing) for details.

## MIT License

As it should be!
