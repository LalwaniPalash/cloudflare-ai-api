# Cloudflare AI API

A production-ready API wrapper for Cloudflare Workers AI with 60+ supported models, comprehensive validation, and monitoring.

## Features

- **60+ AI Models**: Text generation, embeddings, images, classification, and speech recognition
- **RESTful API**: Clean endpoints with consistent response format
- **Authentication**: Bearer token security
- **Monitoring**: Request tracking, processing time, and structured logging
- **Documentation**: Interactive API docs at `/docs`

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/LalwaniPalash/cloudflare-ai-api.git
cd cloudflare-ai-api
npm install
```

### 2. Set Up Configuration

Copy the example configuration:
```bash
cp wrangler.jsonc.example wrangler.jsonc
```

Edit `wrangler.jsonc` to set your project name and API key:
```json
{
  "name": "your-project-name",
  "main": "src/index.ts",
  "compatibility_date": "2024-10-22",
  "ai": {
    "binding": "AI"
  },
  "vars": {
    "API_KEY_SECRET": "your-secure-api-key-here"
  }
}
```

### 3. Deploy

```bash
# Deploy to Cloudflare Workers
npm run deploy

# Or run locally for development
npm run dev
```

### 4. Test Your API

```bash
# Health check
curl https://your-worker.your-subdomain.workers.dev/api/v1/health

# Generate text
curl -X POST https://your-worker.your-subdomain.workers.dev/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "prompt": "Write a haiku about AI",
    "model": "@cf/meta/llama-3.1-8b-instruct"
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation |
| `/api/v1/health` | GET | Health check |
| `/api/v1/models` | GET | List all available models |
| `/api/v1/generate` | POST | Generate text |
| `/api/v1/embeddings` | POST | Create text embeddings |
| `/api/v1/images` | POST | Generate images |
| `/api/v1/classify` | POST | Classify text |
| `/api/v1/summarize` | POST | Summarize text |

## Authentication

All API endpoints (except `/docs`) require authentication:

```bash
Authorization: Bearer YOUR_API_KEY_SECRET
```

Set your API key using Wrangler:
```bash
wrangler secret put API_KEY_SECRET
```

## Supported Models

### Text Generation
- `@cf/openai/gpt-oss-120b`
- `@cf/openai/gpt-oss-20b`
- `@cf/meta/llama-3.1-8b-instruct` (default)
- And 34 more models

### Other Categories
- **Embeddings**: 7 models including BGE series
- **Image Generation**: 8 models including Stable Diffusion and Flux
- **Classification**: 3 models including DistilBERT
- **Speech Recognition**: 5 models including Whisper

See full list at `/api/v1/models`

## Example Usage

### Text Generation

```json
POST /api/v1/generate
{
  "prompt": "Explain quantum computing",
  "model": "@cf/meta/llama-3.1-8b-instruct",
  "max_tokens": 500,
  "temperature": 0.7
}
```

### Image Generation

```json
POST /api/v1/images
{
  "prompt": "A futuristic city at sunset",
  "model": "@cf/bytedance/stable-diffusion-xl-lightning",
  "width": 1024,
  "height": 1024
}
```

## Development

```bash
# Install dependencies
npm install

# Type checking
npm run typecheck

# Run tests
npm test

# Start development server
npm run dev

# View docs locally
open http://localhost:8787/docs
```

## Response Format

All responses follow this structure:

```json
{
  "success": true,
  "data": { /* AI response */ },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "processingTime": 1250
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## License

MIT License - see [LICENSE](./LICENSE) file for details.

## Repository

https://github.com/LalwaniPalash/cloudflare-ai-api