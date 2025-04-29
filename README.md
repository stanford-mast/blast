<div align="center">
  <img src="assets/blast_icon_only.svg" width="200" height="200" alt="BLAST Logo">
</div>

<h1 align="center">BLAST: Browser Language Agent System for Tasks 🚀</h1>

<div align="center">

[![Website](https://img.shields.io/badge/Website-blastproject.org-FFE067)](https://blastproject.org)
[![Documentation](https://img.shields.io/badge/Docs-docs.blastproject.org-FFE067)](https://docs.blastproject.org)
[![Twitter Follow](https://img.shields.io/twitter/follow/realcalebwin?style=social)](https://x.com/realcalebwin)

</div>

BLAST is a high-performance framework for building AI applications with browser automation. It provides a simple OpenAI-compatible API with powerful features like streaming, caching, and parallel execution.

## 🎥 See it in Action

Check out our [demo video](https://vimeo.com/1079613095/7e90cc78f7?ts=0&share=copy) to see BLAST's capabilities!

## 🚀 Quick Start

Install BLAST:
```bash
pip install blastai
```

Start the server:
```bash
blastai serve
```

Use it like the OpenAI API:
```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://127.0.0.1:8000"
)

# Stream real-time browser actions
stream = client.responses.create(
    model="not-needed",
    input="Search for Python documentation and click the first result",
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

## ✨ Features

- 🔄 **OpenAI-Compatible API** - Drop-in replacement for OpenAI's API
- 🚄 **High Performance** - Built for concurrency and parallel processing
- 📡 **Streaming Support** - Real-time streaming of AI responses
- 🔧 **Flexible Engine** - Customizable engine for different AI models
- 💾 **Smart Caching** - Intelligent caching system for improved performance
- 📊 **Resource Management** - Efficient handling of compute resources

## 📚 Documentation

Visit our [documentation](https://docs.blastproject.org) to learn more about:
- Getting started guide
- API reference
- Advanced features
- Development guide

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://docs.blastproject.org/development/contributing) for details.

## 📄 License

BLAST is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <a href="https://blastproject.org">Website</a>
  ·
  <a href="https://docs.blastproject.org">Documentation</a>
  ·
  <a href="https://x.com/realcalebwin">Twitter</a>
  ·
  <a href="https://github.com/stanford-mast/blast">GitHub</a>
</div>