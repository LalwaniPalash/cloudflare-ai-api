import { generateDocumentation, generateHTMLDocumentation, generateMarkdownDocumentation } from './docs'
export interface Env {
  AI: Ai
  API_KEY_SECRET: string
}

interface APIResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  metadata?: {
    model?: string
    processingTime?: number
    timestamp: string
    requestId: string
    tokensUsed?: number
  }
}

interface TextGenerationRequest {
  prompt?: string
  system?: string
  user?: string
  model?: string
  max_tokens?: number
  temperature?: number
  top_p?: number
  stream?: boolean
  instructions?: string
  input?: string
  reasoning?: {
    effort?: string
    summary?: string
  }
}

interface EmbeddingRequest {
  input: string | string[]
  model?: string
}

interface ImageGenerationRequest {
  prompt: string
  model?: string
  num_steps?: number
  guidance?: number
  width?: number
  height?: number
}

interface ClassificationRequest {
  text: string
  categories?: string[]
  model?: string
}

interface SummarizationRequest {
  text?: string
  system?: string
  user?: string
  style?: string
  max_length?: number
}

const SUPPORTED_MODELS = {
  'text-generation': {
    models: [
      '@cf/openai/gpt-oss-120b',
      '@cf/openai/gpt-oss-20b',
      '@cf/meta/llama-4-scout-17b-16e-instruct',
      '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
      '@cf/meta/llama-3.1-8b-instruct-fast',
      '@cf/ibm/granite-4.0-h-micro',
      '@cf/gemma/gemma-sea-lion-v4-27b-it',
      '@cf/google/gemma-3-12b-it',
      '@cf/mistralai/mistral-small-3.1-24b-instruct',
      '@cf/qwen/qwq-32b',
      '@cf/qwen/qwen2.5-coder-32b-instruct',
      '@cf/meta/llama-guard-3-8b',
      '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
            '@cf/meta/llama-3.2-1b-instruct',
      '@cf/meta/llama-3.2-3b-instruct',
      '@cf/meta/llama-3.2-11b-vision-instruct',
      '@cf/meta/llama-3.1-8b-instruct-awq',
      '@cf/meta/llama-3.1-8b-instruct-fp8',
      '@cf/meta/llama-3.1-8b-instruct',
      '@cf/meta/llama-3-8b-instruct-awq',
      '@cf/meta/llama-3-8b-instruct',
      '@cf/microsoft/phi-2',
      '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
      '@cf/qwen/qwen1.5-14b-chat-awq',
      '@cf/qwen/qwen1.5-7b-chat-awq',
      '@cf/qwen/qwen1.5-0.5b-chat',
      '@cf/thebloke/discolm-german-7b-v1-awq',
      '@cf/tiiuae/falcon-7b-instruct',
      '@cf/openchat/openchat-3.5-0106',
      '@cf/defog/sqlcoder-7b-2',
      '@cf/deepseek-ai/deepseek-math-7b-instruct',
      '@cf/nexusflow/starling-lm-7b-beta',
      '@cf/nousresearch/hermes-2-pro-mistral-7b',
      '@cf/mistralai/mistral-7b-instruct-v0.2-lora',
      '@cf/qwen/qwen1.5-1.8b-chat'
    ],
    defaultModel: '@cf/meta/llama-3.1-8b-instruct' as const,
    parameters: {
      max_tokens: { min: 1, max: 8192, default: 512 },
      temperature: { min: 0, max: 2, default: 0.7 },
      top_p: { min: 0, max: 1, default: 1 }
    }
  },
  'text-embeddings': {
    models: [
      '@cf/pfnet/plamo-embedding-1b',
      '@cf/google/embeddinggemma-300m',
      '@cf/baai/bge-reranker-base',
      '@cf/baai/bge-m3',
      '@cf/baai/bge-large-en-v1.5',
      '@cf/baai/bge-small-en-v1.5',
      '@cf/baai/bge-base-en-v1.5'
    ],
    defaultModel: '@cf/baai/bge-base-en-v1.5' as const
  },
  'text-to-image': {
    models: [
      '@cf/leonardo/lucid-origin',
      '@cf/leonardo/phoenix-1.0',
      '@cf/blackforestlabs/flux-1-schnell',
      '@cf/bytedance/stable-diffusion-xl-lightning',
      '@cf/lykon/dreamshaper-8-lcm',
      '@cf/runwayml/stable-diffusion-v1-5-img2img',
      '@cf/runwayml/stable-diffusion-v1-5-inpainting',
      '@cf/stabilityai/stable-diffusion-xl-base-1.0'
    ],
    defaultModel: '@cf/bytedance/stable-diffusion-xl-lightning' as const,
    parameters: {
      num_steps: { min: 1, max: 50, default: 20 },
      guidance: { min: 1, max: 20, default: 7.5 },
      width: { min: 256, max: 1024, default: 512 },
      height: { min: 256, max: 1024, default: 512 }
    }
  },
  'text-classification': {
    models: [
      '@cf/baai/bge-reranker-base',
      '@cf/huggingface/distilbert-sst-2-int8',
      '@cf/meta/llama-guard-3-8b'
    ],
    defaultModel: '@cf/huggingface/distilbert-sst-2-int8' as const
  },
  'automatic-speech-recognition': {
    models: [
      '@cf/deepgram/flux',
      '@cf/deepgram/nova-3',
      '@cf/openai/whisper-large-v3-turbo',
      '@cf/openai/whisper-tiny-en',
      '@cf/openai/whisper'
    ],
    defaultModel: '@cf/openai/whisper' as const
  }
} as const

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

function validateModel(taskType: keyof typeof SUPPORTED_MODELS, modelName: string): boolean {
  const task = SUPPORTED_MODELS[taskType] as any
  return task?.models.includes(modelName) || false
}

function validateParameters(taskType: keyof typeof SUPPORTED_MODELS, params: any): { valid: boolean, errors: string[] } {
  const errors: string[] = []
  const task = SUPPORTED_MODELS[taskType] as any
  
  if (!task?.parameters) return { valid: true, errors: [] }
  
  for (const [param, config] of Object.entries(task.parameters) as [string, any][]) {
    if (params[param] !== undefined) {
      const value = params[param]
      if (typeof value === 'number') {
        if (value < config.min || value > config.max) {
          errors.push(`${param} must be between ${config.min} and ${config.max}`)
        }
      }
    }
  }
  
  return { valid: errors.length === 0, errors }
}

function createResponse<T>(data: T, metadata: any = {}): APIResponse<T> {
  return {
    success: true,
    data,
    metadata: {
      timestamp: new Date().toISOString(),
      ...metadata
    }
  }
}

function createErrorResponse(error: string, metadata: any = {}): APIResponse {
  return {
    success: false,
    error,
    metadata: {
      timestamp: new Date().toISOString(),
      ...metadata
    }
  }
}

function logRequest(requestId: string, method: string, path: string, startTime: number) {
  const duration = Date.now() - startTime
  console.log(JSON.stringify({
    requestId,
    method,
    path,
    duration,
    timestamp: new Date().toISOString()
  }))
}

async function handleDocs(request: Request): Promise<Response> {
  const url = new URL(request.url)
  const baseUrl = `${url.protocol}//${url.host}`

  const acceptHeader = request.headers.get('Accept') || ''
  const docs = generateDocumentation(baseUrl)

  if (acceptHeader.includes('text/markdown')) {
    const markdown = generateMarkdownDocumentation(docs)
    return new Response(markdown, {
      headers: {
        'Content-Type': 'text/markdown',
        'Access-Control-Allow-Origin': '*'
      }
    })
  } else if (acceptHeader.includes('application/json')) {
    return Response.json(docs, {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    })
  } else {
    const html = generateHTMLDocumentation(docs)
    return new Response(html, {
      headers: {
        'Content-Type': 'text/html',
        'Access-Control-Allow-Origin': '*'
      }
    })
  }
}

async function handleHealth(env: Env, requestId: string): Promise<Response> {
  const startTime = Date.now()
  
  try {
    await env.AI.run(SUPPORTED_MODELS['text-generation'].defaultModel as any, { 
      prompt: 'Hello', 
      max_tokens: 1 
    })
    
    return Response.json(createResponse({
      status: 'healthy',
      services: {
        ai: 'operational'
      },
      version: '1.0.0'
    }, { requestId, processingTime: Date.now() - startTime }))
    
  } catch (error) {
    return Response.json(createErrorResponse(
      'AI service unavailable',
      { requestId, processingTime: Date.now() - startTime }
    ), { status: 503 })
  }
}

async function handleModels(requestId: string): Promise<Response> {
  const models = {
    'text-generation': SUPPORTED_MODELS['text-generation'],
    'text-embeddings': SUPPORTED_MODELS['text-embeddings'],
    'text-to-image': SUPPORTED_MODELS['text-to-image'],
    'text-classification': SUPPORTED_MODELS['text-classification'],
    'automatic-speech-recognition': SUPPORTED_MODELS['automatic-speech-recognition']
  }
  
  return Response.json(createResponse({
    models,
    total: Object.values(SUPPORTED_MODELS).reduce((acc, task) => acc + task.models.length, 0)
  }, { requestId }))
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const startTime = Date.now()
    const requestId = generateRequestId()
    const url = new URL(request.url)
    const path = url.pathname

    try {
      if (request.method === 'OPTIONS') {
        return new Response(null, {
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400'
          }
        })
      }

      if (path.startsWith('/api/')) {
        const authHeader = request.headers.get('Authorization')
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
          return Response.json(createErrorResponse('Authorization header required', { requestId }), { status: 401 })
        }
        
        const token = authHeader.substring(7)
        if (token !== env.API_KEY_SECRET) {
          return Response.json(createErrorResponse('Invalid API key', { requestId }), { status: 401 })
        }
      }

      let response: Response
      
      switch (true) {
        case path === '/' && request.method === 'GET':
          return Response.redirect(new URL('/docs', request.url).toString(), 302)
          
        case path === '/docs' && request.method === 'GET':
          response = await handleDocs(request)
          break
          
        case path === '/api/v1/health' && request.method === 'GET':
          response = await handleHealth(env, requestId)
          break
          
        case path === '/api/v1/models' && request.method === 'GET':
          response = await handleModels(requestId)
          break
          
        case path === '/api/v1/generate' && request.method === 'POST':
          response = await handleTextGeneration(request, env, requestId)
          break
          
        case path === '/api/v1/embeddings' && request.method === 'POST':
          response = await handleTextEmbedding(request, env, requestId)
          break
          
        case path === '/api/v1/images' && request.method === 'POST':
          response = await handleImageGeneration(request, env, requestId)
          break
          
        case path === '/api/v1/classify' && request.method === 'POST':
          response = await handleTextClassification(request, env, requestId)
          break
          
        case path === '/api/v1/summarize' && request.method === 'POST':
          response = await handleSummarization(request, env, requestId)
          break
          
        default:
          response = Response.json(
            createErrorResponse('Endpoint not found', { requestId }), 
            { status: 404 }
          )
      }

      response.headers.set('Access-Control-Allow-Origin', '*')
      response.headers.set('X-Request-ID', requestId)
      response.headers.set('X-Processing-Time', `${Date.now() - startTime}ms`)
      
      logRequest(requestId, request.method, path, startTime)
      return response
      
    } catch (error) {
      console.error('Unhandled error:', error)
      return Response.json(
        createErrorResponse(
          error instanceof Error ? error.message : 'Internal server error', 
          { requestId }
        ),
        { status: 500 }
      )
    }
  }
}

async function handleTextGeneration(
  request: Request, 
  env: Env, 
  requestId: string
): Promise<Response> {
  const startTime = Date.now()
  
  try {
    const body: TextGenerationRequest = await request.json()
    
    let finalPrompt: string
    if (body.system && body.user) {
      finalPrompt = `<system>${body.system}</system>\n\n<user>${body.user}</user>`
    } else if (body.prompt) {
      finalPrompt = body.prompt
    } else if (body.user) {
      finalPrompt = body.user
    } else {
      return Response.json(
        createErrorResponse('Either "prompt" or "user" field is required', { requestId }),
        { status: 400 }
      )
    }
    
    const model = body.model || SUPPORTED_MODELS['text-generation'].defaultModel
    
    if (!validateModel('text-generation', model)) {
      return Response.json(
        createErrorResponse(`Unsupported model: ${model}`, { requestId }),
        { status: 400 }
      )
    }
    
    const validation = validateParameters('text-generation', body)
    if (!validation.valid) {
      return Response.json(
        createErrorResponse(`Parameter validation failed: ${validation.errors.join(', ')}`, { requestId }),
        { status: 400 }
      )
    }
    
    let aiParams: any

    if (model.includes('gpt-oss')) {
      aiParams = {
        input: body.user || body.prompt || finalPrompt,
        max_tokens: body.max_tokens || 512
      }

      if (body.system) {
        aiParams.instructions = body.system
      }
      if (body.reasoning) {
        aiParams.reasoning = body.reasoning
      }
    } else {
      aiParams = {
        prompt: finalPrompt,
        max_tokens: body.max_tokens || 512,
        temperature: body.temperature || 0.7,
        top_p: body.top_p || 1,
        stream: body.stream || false
      }
    }

    const aiResponse = await env.AI.run(model as any, aiParams)
    
    return Response.json(createResponse(aiResponse, {
      requestId,
      model,
      processingTime: Date.now() - startTime,
      tokensUsed: (aiResponse as any)?.meta?.tokens_used
    }))
    
  } catch (error) {
    return Response.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Text generation failed',
        { requestId, processingTime: Date.now() - startTime }
      ),
      { status: 500 }
    )
  }
}

async function handleTextEmbedding(
  request: Request,
  env: Env,
  requestId: string
): Promise<Response> {
  const startTime = Date.now()
  
  try {
    const body: EmbeddingRequest = await request.json()
    
    if (!body.input) {
      return Response.json(
        createErrorResponse('Input is required', { requestId }),
        { status: 400 }
      )
    }
    
    const model = body.model || SUPPORTED_MODELS['text-embeddings'].defaultModel
    
    if (!validateModel('text-embeddings', model)) {
      return Response.json(
        createErrorResponse(`Unsupported embedding model: ${model}`, { requestId }),
        { status: 400 }
      )
    }
    
    const aiResponse = await env.AI.run(model as any, {
      text: Array.isArray(body.input) ? body.input : [body.input]
    })
    
    return Response.json(createResponse(aiResponse, {
      requestId,
      model,
      processingTime: Date.now() - startTime
    }))
    
  } catch (error) {
    return Response.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Text embedding failed',
        { requestId, processingTime: Date.now() - startTime }
      ),
      { status: 500 }
    )
  }
}

async function handleImageGeneration(
  request: Request,
  env: Env,
  requestId: string
): Promise<Response> {
  const startTime = Date.now()
  
  try {
    const body: ImageGenerationRequest = await request.json()
    
    if (!body.prompt) {
      return Response.json(
        createErrorResponse('Prompt is required', { requestId }),
        { status: 400 }
      )
    }
    
    const model = body.model || SUPPORTED_MODELS['text-to-image'].defaultModel
    
    if (!validateModel('text-to-image', model)) {
      return Response.json(
        createErrorResponse(`Unsupported image model: ${model}`, { requestId }),
        { status: 400 }
      )
    }
    
    const validation = validateParameters('text-to-image', body)
    if (!validation.valid) {
      return Response.json(
        createErrorResponse(`Parameter validation failed: ${validation.errors.join(', ')}`, { requestId }),
        { status: 400 }
      )
    }
    
    let aiResponse: any
    
    try {
      aiResponse = await env.AI.run(model as any, {
        prompt: body.prompt,
        num_steps: body.num_steps || 20,
        guidance: body.guidance || 7.5,
        width: body.width || 512,
        height: body.height || 512
      })
    } catch (error) {
      return Response.json(
        createErrorResponse(
          `AI model error: ${error instanceof Error ? error.message : String(error)}`,
          { requestId, processingTime: Date.now() - startTime }
        ),
        { status: 500 }
      )
    }
    
    // Handle the ReadableStream response from Cloudflare Workers AI
    let processedResponse: any
    
    // Check if it's a ReadableStream (expected for image generation)
    if (aiResponse instanceof ReadableStream) {
      try {
        // Convert ReadableStream to ArrayBuffer using Response
        const response = new Response(aiResponse)
        const arrayBuffer = await response.arrayBuffer()
        
        // Convert ArrayBuffer to base64 using a more efficient method
        const bytes = new Uint8Array(arrayBuffer)
        let binaryString = ''
        
        // Process in smaller chunks to avoid stack overflow
        const chunkSize = 8192 // 8KB chunks
        for (let i = 0; i < bytes.length; i += chunkSize) {
          const end = Math.min(i + chunkSize, bytes.length)
          const chunk = bytes.slice(i, end)
          binaryString += String.fromCharCode(...chunk)
        }
        
        const base64String = btoa(binaryString)
        
        processedResponse = {
          image: base64String,
          format: "png",
          size: arrayBuffer.byteLength,
          contentType: "image/png"
        }
      } catch (error) {
        processedResponse = {
          error: "Failed to process ReadableStream",
          details: error instanceof Error ? error.message : String(error)
        }
      }
    }
    // Check if it's a Response object
    else if (aiResponse instanceof Response) {
      const contentType = aiResponse.headers.get('content-type') || 'unknown'
      
      if (contentType.includes('image/')) {
        const arrayBuffer = await aiResponse.arrayBuffer()
        const uint8Array = new Uint8Array(arrayBuffer)
        const base64String = btoa(String.fromCharCode(...uint8Array))
        
        processedResponse = {
          image: base64String,
          format: contentType.split('/')[1] || 'png',
          size: arrayBuffer.byteLength,
          contentType: contentType
        }
      } else {
        const text = await aiResponse.text()
        processedResponse = {
          rawText: text,
          contentType: contentType,
          note: "Response was not an image"
        }
      }
    }
    // Check for direct binary data
    else if (aiResponse instanceof ArrayBuffer) {
      const uint8Array = new Uint8Array(aiResponse)
      const base64String = btoa(String.fromCharCode(...uint8Array))
      processedResponse = {
        image: base64String,
        format: "png",
        size: aiResponse.byteLength
      }
    }
    else if (aiResponse instanceof Uint8Array) {
      const base64String = btoa(String.fromCharCode(...aiResponse))
      processedResponse = {
        image: base64String,
        format: "png", 
        size: aiResponse.length
      }
    }
    // Handle other object types (for debugging)
    else if (typeof aiResponse === 'object' && aiResponse !== null) {
      const keys = Object.keys(aiResponse)
      processedResponse = {
        debugInfo: {
          responseType: typeof aiResponse,
          responseKeys: keys,
          constructor: aiResponse.constructor.name,
          rawResponse: keys.length === 0 ? aiResponse : "Has content but unknown format"
        },
        note: "Unknown object format - expected ReadableStream for image generation"
      }
    }
    else {
      processedResponse = {
        debugInfo: {
          responseType: typeof aiResponse,
          rawResponse: aiResponse
        },
        note: "Unexpected response format - expected ReadableStream"
      }
    }
    
    return Response.json(createResponse(processedResponse, {
      requestId,
      model,
      processingTime: Date.now() - startTime
    }))
    
  } catch (error) {
    return Response.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Image generation failed',
        { requestId, processingTime: Date.now() - startTime }
      ),
      { status: 500 }
    )
  }
}

async function handleTextClassification(
  request: Request,
  env: Env,
  requestId: string
): Promise<Response> {
  const startTime = Date.now()
  
  try {
    const body: ClassificationRequest = await request.json()
    
    if (!body.text) {
      return Response.json(
        createErrorResponse('Text is required', { requestId }),
        { status: 400 }
      )
    }
    
    const model = body.model || SUPPORTED_MODELS['text-classification'].defaultModel
    
    if (!validateModel('text-classification', model)) {
      return Response.json(
        createErrorResponse(`Unsupported classification model: ${model}`, { requestId }),
        { status: 400 }
      )
    }
    
    const aiResponse = await env.AI.run(model as any, {
      text: body.text,
      labels: body.categories || []
    })
    
    return Response.json(createResponse(aiResponse, {
      requestId,
      model,
      processingTime: Date.now() - startTime
    }))
    
  } catch (error) {
    return Response.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Text classification failed',
        { requestId, processingTime: Date.now() - startTime }
      ),
      { status: 500 }
    )
  }
}

async function handleSummarization(
  request: Request,
  env: Env,
  requestId: string
): Promise<Response> {
  const startTime = Date.now()
  
  try {
    const body: SummarizationRequest = await request.json()
    
    // Handle different input formats
    let finalPrompt: string
    if (body.system && body.user) {
      // System + User format
      finalPrompt = `<system>${body.system}</system>\n\n<user>${body.user}</user>`
    } else if (body.text) {
      // Legacy text format
      const summaryPrompt = `Please summarize the following text in a ${body.style || 'concise'} manner:\n\n${body.text}\n\nSummary:`
      finalPrompt = summaryPrompt
    } else if (body.user) {
      // User only format with default system prompt
      const defaultSystem = `You are a helpful assistant that provides ${body.style || 'concise'} summaries of text.`
      finalPrompt = `<system>${defaultSystem}</system>\n\n<user>${body.user}</user>`
    } else {
      return Response.json(
        createErrorResponse('Either "text" or "user" field is required', { requestId }),
        { status: 400 }
      )
    }
    
    // Use text generation for summarization with the constructed prompt
    const model = SUPPORTED_MODELS['text-generation'].defaultModel
    
    const aiResponse = await env.AI.run(model as any, {
      prompt: finalPrompt,
      max_tokens: body.max_length || 256,
      temperature: 0.3,
      top_p: 0.9
    })
    
    return Response.json(createResponse(aiResponse, {
      requestId,
      model,
      processingTime: Date.now() - startTime,
      tokensUsed: (aiResponse as any)?.meta?.tokens_used
    }))
    
  } catch (error) {
    return Response.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Text summarization failed',
        { requestId, processingTime: Date.now() - startTime }
      ),
      { status: 500 }
    )
  }
}
