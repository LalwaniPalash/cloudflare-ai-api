# Enhanced Cloudflare Workers AI API

A comprehensive, production-ready API for Cloudflare Workers AI with advanced features including model validation, request handling, monitoring, and structured logging. Supports 25+ AI models across text generation, embeddings, image generation, classification, and speech recognition.

**Version:** 1.0.0
**Contact:** [API Support](https://github.com/your-repo/personal-ai)

## 🔐 Authentication

**Type:** Bearer Token

Most endpoints require authentication using a Bearer token in the Authorization header. The /docs endpoint is public.

**Header:** `Authorization: Bearer your-api-key-here`

## 🚀 API Endpoints

### Table of Contents

1. [GET /docs](#get--docs) - API Documentation
2. [GET /api/v1/health](#get--api-v1-health) - Health Check
3. [GET /api/v1/models](#get--api-v1-models) - List Available Models
4. [POST /api/v1/generate](#post--api-v1-generate) - Text Generation
5. [POST /api/v1/embeddings](#post--api-v1-embeddings) - Text Embeddings
6. [POST /api/v1/images](#post--api-v1-images) - Image Generation
7. [POST /api/v1/classify](#post--api-v1-classify) - Text Classification
8. [POST /api/v1/summarize](#post--api-v1-summarize) - Text Summarization


### GET /docs

Returns comprehensive API documentation including all endpoints, schemas, and examples. This endpoint is public and does not require authentication.

**Authentication:** 🌐 Public

**Request Example:**
```bash
curl -X GET https://personal-ai.palashlalwani-r.workers.dev/docs
```

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "info": {
    "title": "Enhanced Cloudflare Workers AI API",
    "version": "1.0.0"
  },
  "endpoints": [
    "..."
  ],
  "models": {
    "text-generation": [
      "..."
    ]
  }
}
```

---

### GET /api/v1/health

Checks the health status of the API and AI service connectivity. Performs a lightweight test of the AI service to ensure it's operational.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X GET https://personal-ai.palashlalwani-r.workers.dev/api/v1/health -H "Authorization: Bearer your-api-key"
```

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "services": {
      "ai": "operational"
    },
    "version": "1.0.0"
  },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "processingTime": 1250
  }
}
```

---

### GET /api/v1/models

Returns all available AI models organized by category, including their capabilities, parameters, and default configurations.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X GET https://personal-ai.palashlalwani-r.workers.dev/api/v1/models -H "Authorization: Bearer your-api-key"
```

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "models": {
      "text-generation": {
        "models": [
          "@cf/openai/gpt-oss-120b",
          "@cf/openai/gpt-oss-20b",
          "@cf/meta/llama-4-scout-17b-16e-instruct"
        ],
        "defaultModel": "@cf/meta/llama-3.1-8b-instruct",
        "parameters": {
          "max_tokens": {
            "min": 1,
            "max": 8192,
            "default": 512
          },
          "temperature": {
            "min": 0,
            "max": 2,
            "default": 0.7
          }
        }
      }
    },
    "total": 17
  }
}
```

---

### POST /api/v1/generate

Generates text using large language models. Supports both legacy single prompt format and new system+user prompt format for enhanced control over AI behavior and responses.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "system": "You are a helpful assistant that explains complex topics in simple terms.",
    "user": "What is machine learning?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Request Body:**
```json
{
  "system": "You are a helpful assistant that explains complex topics in simple terms.",
  "user": "What is machine learning?",
  "max_tokens": 150,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Machine learning is a way for computers to learn and make decisions by looking at lots of examples, just like how you learn to recognize cats by seeing many different cats.",
    "usage": {
      "prompt_tokens": 42,
      "completion_tokens": 28,
      "total_tokens": 70
    }
  },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "processingTime": 1500,
    "tokensUsed": 28
  }
}
```

---

### POST /api/v1/embeddings

Generates vector embeddings for text input. Useful for semantic search, similarity comparisons, and other NLP tasks.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "input": "Artificial intelligence is transforming the world"
  }'
```

**Request Body:**
```json
{
  "input": "Artificial intelligence is transforming the world",
  "model": "@cf/baai/bge-base-en-v1.5"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "data": [
      [
        0.123,
        -0.456,
        0.789,
        "... (768 dimensions)"
      ]
    ],
    "shape": [
      1,
      768
    ],
    "usage": {
      "prompt_tokens": 9,
      "total_tokens": 9
    }
  },
  "metadata": {
    "requestId": "req_1704067200000_def456",
    "processingTime": 890
  }
}
```

---

### POST /api/v1/images

Generates high-quality images from text prompts using Stable Diffusion models. Returns base64-encoded image data ready for saving as PNG/JPEG files.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/images \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A beautiful sunset over a mountain landscape, photorealistic, 4k quality",
    "width": 1024,
    "height": 1024,
    "guidance": 7.5,
    "num_steps": 20
  }'
```

**Request Body:**
```json
{
  "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
  "prompt": "A majestic eagle soaring over snowy mountains, highly detailed, award-winning photography",
  "width": 1024,
  "height": 1024,
  "guidance": 7.5,
  "num_steps": 20
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOt...",
    "format": "png",
    "size": 306108,
    "contentType": "image/png"
  },
  "metadata": {
    "requestId": "req_1759055109536_6sbo5kq0h",
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "processingTime": 4839
  }
}
```

---

### POST /api/v1/classify

Classifies text into predefined categories or performs sentiment analysis.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "text": "I love this new AI service!",
    "categories": ["positive", "negative", "neutral"]
  }'
```

**Request Body:**
```json
{
  "text": "I love this new AI service! It's incredibly helpful.",
  "categories": [
    "positive",
    "negative",
    "neutral"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "label": "POSITIVE",
      "score": 0.9987
    },
    {
      "label": "NEGATIVE",
      "score": 0.0013
    }
  ],
  "metadata": {
    "requestId": "req_1704067200000_jkl012",
    "processingTime": 1200
  }
}
```

---

### POST /api/v1/summarize

Summarizes long text into shorter, more concise versions while preserving key information. Supports both legacy text format and new system+user prompt format.

**Authentication:** 🔒 Required

**Request Example:**
```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/summarize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "system": "You are a professional summarizer.",
    "user": "Summarize this: [long text here]",
    "max_length": 100
  }'
```

**Request Body:**
```json
{
  "system": "You are a professional summarizer who creates clear, concise summaries that capture the most important points.",
  "user": "Summarize this research paper: Artificial intelligence has made significant advances in natural language processing, computer vision, and robotics. Recent developments include more efficient training methods, better model architectures, and improved performance on benchmark tasks.",
  "max_length": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "AI has advanced significantly in NLP, computer vision, and robotics through improved training methods, model architectures, and benchmark performance.",
    "usage": {
      "prompt_tokens": 89,
      "completion_tokens": 23,
      "total_tokens": 112
    }
  },
  "metadata": {
    "requestId": "req_1704067200000_mno345",
    "processingTime": 1800
  }
}
```


## 🤖 Available Models


### Text Generation

Large language models for text generation, completion, and conversation

**Default Model:** `@cf/meta/llama-3.1-8b-instruct`

**Available Models:**
- `@cf/openai/gpt-oss-120b`
- `@cf/openai/gpt-oss-20b`
- `@cf/meta/llama-4-scout-17b-16e-instruct`
- `@cf/meta/llama-3.3-70b-instruct-fp8-fast`
- `@cf/meta/llama-3.1-8b-instruct-fast`
- `@cf/ibm/granite-4.0-h-micro`
- `@cf/gemma/gemma-sea-lion-v4-27b-it`
- `@cf/google/gemma-3-12b-it`
- `@cf/mistralai/mistral-small-3.1-24b-instruct`
- `@cf/qwen/qwq-32b`
- `@cf/qwen/qwen2.5-coder-32b-instruct`
- `@cf/meta/llama-guard-3-8b`
- `@cf/deepseek/deepseek-r1-distill-qwen-32b`
- `@cf/meta/llama-3.2-1b-instruct`
- `@cf/meta/llama-3.2-3b-instruct`
- `@cf/meta/llama-3.2-11b-vision-instruct`
- `@cf/meta/llama-3.1-8b-instruct-awq`
- `@cf/meta/llama-3.1-8b-instruct-fp8`
- `@cf/meta/llama-3.1-8b-instruct`
- `@cf/meta/meta-llama-3-8b-instruct`
- `@cf/meta/llama-3-8b-instruct-awq`
- `@cf/meta/llama-3-8b-instruct`
- `@cf/microsoft/phi-2`
- `@cf/tinyllama/tinyllama-1.1b-chat-v1.0`
- `@cf/qwen/qwen1.5-14b-chat-awq`
- `@cf/qwen/qwen1.5-7b-chat-awq`
- `@cf/qwen/qwen1.5-0.5b-chat`
- `@cf/thebloke/discolm-german-7b-v1-awq`
- `@cf/tiiuae/falcon-7b-instruct`
- `@cf/openchat/openchat-3.5-0106`
- `@cf/defog/sqlcoder-7b-2`
- `@cf/deepseek/deepseek-math-7b-instruct`
- `@cf/nexusflow/starling-lm-7b-beta`
- `@cf/nousresearch/hermes-2-pro-mistral-7b`
- `@cf/mistralai/mistral-7b-instruct-v0.2-lora`
- `@cf/qwen/qwen1.5-1.8b-chat`


**Parameters:**
- `max_tokens`: Maximum tokens to generate (min: 1, max: 8192, default: 512)
- `temperature`: Randomness control (0=deterministic) (min: N/A, max: 2, default: 0.7)
- `top_p`: Nucleus sampling parameter (min: N/A, max: 1, default: 1)



### Text Embeddings

Models for converting text into vector embeddings for semantic search and similarity

**Default Model:** `@cf/baai/bge-base-en-v1.5`

**Available Models:**
- `@cf/pfnet/plamo-embedding-1b`
- `@cf/google/embeddinggemma-300m`
- `@cf/baai/bge-reranker-base`
- `@cf/baai/bge-m3`
- `@cf/baai/bge-large-en-v1.5`
- `@cf/baai/bge-small-en-v1.5`
- `@cf/baai/bge-base-en-v1.5`




### Text To Image

Stable Diffusion models for generating high-quality images from text prompts. Returns base64-encoded PNG/JPEG data ready for saving. stable-diffusion-xl-base-1.0: High quality, detailed images (4-6s). xl-lightning: Fast generation, good quality (2-4s).

**Default Model:** `@cf/bytedance/stable-diffusion-xl-lightning`

**Available Models:**
- `@cf/leonardo/lucid-origin`
- `@cf/leonardo/phoenix-1.0`
- `@cf/blackforestlabs/flux-1-schnell`
- `@cf/bytedance/stable-diffusion-xl-lightning`
- `@cf/lykon/dreamshaper-8-lcm`
- `@cf/runwayml/stable-diffusion-v1-5-img2img`
- `@cf/runwayml/stable-diffusion-v1-5-inpainting`
- `@cf/stabilityai/stable-diffusion-xl-base-1.0`


**Parameters:**
- `num_steps`: Number of diffusion steps (higher = better quality) (min: 1, max: 20, default: 20)
- `guidance`: Prompt adherence strength (7.5-15 recommended) (min: 1, max: 20, default: 7.5)
- `width`: Image width in pixels (min: 256, max: 2048, default: 512)
- `height`: Image height in pixels (min: 256, max: 2048, default: 512)
- `negative_prompt`: Elements to avoid (e.g., 'blurry, low quality') (min: N/A, max: N/A, default: N/A)
- `seed`: Random seed for reproducible results (min: N/A, max: N/A, default: N/A)



### Text Classification

Models for categorizing text and sentiment analysis

**Default Model:** `@cf/huggingface/distilbert-sst-2-int8`

**Available Models:**
- `@cf/baai/bge-reranker-base`
- `@cf/huggingface/distilbert-sst-2-int8`
- `@cf/meta/llama-guard-3-8b`




### Automatic Speech Recognition

Models for converting speech audio to text

**Default Model:** `@cf/openai/whisper`

**Available Models:**
- `@cf/deepgram/flux`
- `@cf/deepgram/nova-3`
- `@cf/openai/whisper-large-v3-turbo`
- `@cf/openai/whisper-tiny-en`
- `@cf/openai/whisper`




## ⚠️ Error Codes


### 400

Bad Request - Invalid input, missing required fields, or parameter validation failed

**Example:**
```json
{
  "success": false,
  "error": "Parameter validation failed: max_tokens must be between 1 and 8192",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_error1"
  }
}
```


### 401

Unauthorized - Missing or invalid API key

**Example:**
```json
{
  "success": false,
  "error": "Invalid API key",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_error2"
  }
}
```


### 404

Not Found - Endpoint does not exist

**Example:**
```json
{
  "success": false,
  "error": "Endpoint not found",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_error3"
  }
}
```


### 500

Internal Server Error - Unexpected error occurred

**Example:**
```json
{
  "success": false,
  "error": "Internal server error",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_error4"
  }
}
```


### 503

Service Unavailable - AI service is temporarily unavailable

**Example:**
```json
{
  "success": false,
  "error": "AI service unavailable",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_error5"
  }
}
```


## 💡 Usage Examples

### 🎨 Image Generation

Generate high-quality images from text prompts:

```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/images \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A beautiful sunset over ocean waves, vibrant colors, peaceful",
    "width": 1024,
    "height": 1024,
    "guidance": 7.5,
    "num_steps": 20
  }'
```

### 💭 System + User Prompts

The API supports separate system and user prompts for enhanced control:

```bash
curl -X POST https://personal-ai.palashlalwani-r.workers.dev/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "system": "You are a helpful coding assistant. Provide clear, concise answers.",
    "user": "How do I create a REST API in Python?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### 🖼️ Save Generated Image (Python)

```python
import requests
import base64

response = requests.post('https://personal-ai.palashlalwani-r.workers.dev/api/v1/images',
    headers={
        'Authorization': 'Bearer your-api-key',
        'Content-Type': 'application/json'
    },
    json={
        'model': '@cf/bytedance/stable-diffusion-xl-lightning',
        'prompt': 'A beautiful sunset over ocean waves, vibrant colors, peaceful',
        'width': 1024,
        'height': 1024
    }
)

if response.json()['success']:
    data = response.json()['data']
    image_data = base64.b64decode(data['image'])

    with open('generated_image.png', 'wb') as f:
        f.write(image_data)

    print(f"Image saved! Size: {data['size']} bytes")
```

## 📊 Response Format

All API responses follow this consistent structure:

```json
{
  "success": true,
  "data": {
    // AI response data
  },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "processingTime": 1250,
    "tokensUsed": 42
  }
}
```

Error responses:
```json
{
  "success": false,
  "error": "Detailed error message",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "processingTime": 50
  }
}
```

## 🔍 Monitoring & Debugging

### Request Tracking
Every request gets a unique ID for debugging across logs and responses.

### Performance Metrics
- Processing time measurement for each request
- Token usage tracking (when available)
- Model performance comparison

### Structured Logging
```json
{
  "requestId": "req_1704067200000_abc123",
  "method": "POST",
  "path": "/api/v1/generate",
  "duration": 1250,
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

## 🔒 Security

- **API Key Authentication**: Bearer token authentication for all API endpoints
- **CORS Configuration**: Properly configured for web applications
- **Input Validation**: Comprehensive parameter validation to prevent abuse
- **Error Handling**: Safe error messages without sensitive information exposure

## 🚦 Rate Limits & Usage

- Follows Cloudflare Workers AI rate limits
- Processing time limits enforced per request
- Parameter bounds validation prevents resource abuse

## 📈 Performance Optimizations

- **Model Validation Caching**: Pre-computed model validation rules
- **Response Streaming**: Support for streaming responses (where applicable)
- **Efficient Error Handling**: Quick validation failures with detailed feedback
- **Metadata Optimization**: Minimal overhead for request tracking

## 🧪 Testing

```bash
# Run type checking
npm run typecheck

# Test API endpoints locally
curl -X GET http://localhost:8787/api/v1/health

# Test with authentication
curl -X POST http://localhost:8787/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"prompt": "Hello, world!"}'
```

---

**Built with ❤️ using Cloudflare Workers AI**
*Last updated: 2025-10-23T19:23:41.001Z*
