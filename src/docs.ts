export interface APIDocumentation {
  info: {
    title: string
    version: string
    description: string
    contact: {
      name: string
      url: string
    }
  }
  servers: Array<{
    url: string
    description: string
  }>
  authentication: {
    type: string
    description: string
    headerName: string
    example: string
  }
  endpoints: Array<{
    method: string
    path: string
    summary: string
    description: string
    authentication: boolean
    requestSchema?: any
    responseSchema: any
    examples: {
      request?: any
      response: any
      curl: string
    }
  }>
  models: {
    [category: string]: {
      description: string
      models: string[]
      defaultModel: string
      parameters?: any
    }
  }
  errorCodes: Array<{
    code: number
    description: string
    example: any
  }>
}

export function generateDocumentation(baseUrl: string): APIDocumentation {
  return {
    info: {
      title: "Enhanced Cloudflare Workers AI API",
      version: "1.0.0",
      description: "A comprehensive, production-ready API for Cloudflare Workers AI with advanced features including model validation, request handling, monitoring, and structured logging. Supports 25+ AI models across text generation, embeddings, image generation, classification, and speech recognition.",
      contact: {
        name: "API Support",
        url: "https://github.com/your-repo/personal-ai"
      }
    },
    servers: [
      {
        url: baseUrl,
        description: "Development server"
      }
    ],
    authentication: {
      type: "Bearer Token",
      description: "Most endpoints require authentication using a Bearer token in the Authorization header. The /docs endpoint is public.",
      headerName: "Authorization",
      example: "Bearer your-api-key-here"
    },
    endpoints: [
      {
        method: "GET",
        path: "/docs",
        summary: "API Documentation",
        description: "Returns comprehensive API documentation including all endpoints, schemas, and examples. This endpoint is public and does not require authentication.",
        authentication: false,
        responseSchema: {
          type: "object",
          properties: {
            info: { type: "object" },
            endpoints: { type: "array" },
            models: { type: "object" },
            examples: { type: "object" }
          }
        },
        examples: {
          response: {
            info: { title: "Enhanced Cloudflare Workers AI API", version: "1.0.0" },
            endpoints: ["..."],
            models: { "text-generation": ["..."] }
          },
          curl: `curl -X GET ${baseUrl}/docs`
        }
      },
      {
        method: "GET",
        path: "/api/v1/health",
        summary: "Health Check",
        description: "Checks the health status of the API and AI service connectivity. Performs a lightweight test of the AI service to ensure it's operational.",
        authentication: true,
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                status: { type: "string", enum: ["healthy", "unhealthy"] },
                services: {
                  type: "object",
                  properties: {
                    ai: { type: "string", enum: ["operational", "degraded", "unavailable"] }
                  }
                },
                version: { type: "string" }
              }
            },
            metadata: {
              type: "object",
              properties: {
                timestamp: { type: "string", format: "date-time" },
                requestId: { type: "string" },
                processingTime: { type: "number" }
              }
            }
          }
        },
        examples: {
          response: {
            success: true,
            data: {
              status: "healthy",
              services: { ai: "operational" },
              version: "1.0.0"
            },
            metadata: {
              timestamp: "2024-01-01T00:00:00.000Z",
              requestId: "req_1704067200000_abc123",
              processingTime: 1250
            }
          },
          curl: `curl -X GET ${baseUrl}/api/v1/health -H "Authorization: Bearer your-api-key"`
        }
      },
      {
        method: "GET",
        path: "/api/v1/models",
        summary: "List Available Models",
        description: "Returns all available AI models organized by category, including their capabilities, parameters, and default configurations.",
        authentication: true,
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                models: {
                  type: "object",
                  additionalProperties: {
                    type: "object",
                    properties: {
                      models: { type: "array", items: { type: "string" } },
                      defaultModel: { type: "string" },
                      parameters: { type: "object" }
                    }
                  }
                },
                total: { type: "number" }
              }
            },
            metadata: { type: "object" }
          }
        },
        examples: {
          response: {
            success: true,
            data: {
              models: {
                "text-generation": {
                  models: ["@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b", "@cf/meta/llama-4-scout-17b-16e-instruct"],
                  defaultModel: "@cf/meta/llama-3.1-8b-instruct",
                  parameters: {
                    max_tokens: { min: 1, max: 8192, default: 512 },
                    temperature: { min: 0, max: 2, default: 0.7 }
                  }
                }
              },
              total: 17
            }
          },
          curl: `curl -X GET ${baseUrl}/api/v1/models -H "Authorization: Bearer your-api-key"`
        }
      },
      {
        method: "POST",
        path: "/api/v1/generate",
        summary: "Text Generation",
        description: "Generates text using large language models. Supports both legacy single prompt format and new system+user prompt format for enhanced control over AI behavior and responses.",
        authentication: true,
        requestSchema: {
          type: "object",
          properties: {
                        system: {
              type: "string",
              description: "System prompt that defines AI behavior, personality, and response style. For GPT-OSS models, this is passed as 'instructions'"
            },
            user: {
              type: "string",
              description: "User's input prompt or question. For GPT-OSS models, this is passed as 'input'"
            },
                        prompt: {
              type: "string",
              description: "Legacy single prompt format (still supported)"
            },
            model: {
              type: "string",
              description: "Model to use for generation. GPT-OSS models include @cf/openai/gpt-oss-120b and @cf/openai/gpt-oss-20b",
              default: "@cf/meta/llama-3.1-8b-instruct"
            },
            max_tokens: {
              type: "number",
              minimum: 1,
              maximum: 8192,
              default: 512,
              description: "Maximum number of tokens to generate"
            },
            temperature: {
              type: "number",
              minimum: 0,
              maximum: 2,
              default: 0.7,
              description: "Controls randomness (0 = deterministic, 2 = very random)"
            },
            top_p: {
              type: "number",
              minimum: 0,
              maximum: 1,
              default: 1,
              description: "Controls diversity via nucleus sampling"
            },
            stream: {
              type: "boolean",
              default: false,
              description: "Whether to stream the response"
            },
                        reasoning: {
              type: "object",
              description: "GPT-OSS specific reasoning parameters for enhanced step-by-step thinking",
              properties: {
                effort: {
                  type: "string",
                  enum: ["low", "medium", "high"],
                  description: "Reasoning effort level (GPT-OSS models only)"
                },
                summary: {
                  type: "string",
                  description: "Optional summary format for reasoning output (GPT-OSS models only)"
                }
              },
              additionalProperties: true
            },
                        instructions: {
              type: "string",
              description: "Alternative to 'system' field for GPT-OSS models (passed directly as instructions)"
            },
            input: {
              type: "string",
              description: "Alternative to 'user' field for GPT-OSS models (passed directly as input)"
            }
          },
          oneOf: [
            { required: ["system", "user"] },
            { required: ["user"] },
            { required: ["prompt"] },
            { required: ["instructions", "input"] },
            { required: ["input"] }
          ]
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              oneOf: [
                {
                                    type: "object",
                  properties: {
                    response: { type: "string", description: "Generated text response" },
                    usage: {
                      type: "object",
                      properties: {
                        prompt_tokens: { type: "number" },
                        completion_tokens: { type: "number" },
                        total_tokens: { type: "number" }
                      }
                    }
                  }
                },
                {
                                    type: "object",
                  properties: {
                    id: { type: "string", description: "Response ID for GPT-OSS models" },
                    created_at: { type: "number", description: "Creation timestamp for GPT-OSS models" },
                    instructions: { type: ["string", "null"], description: "Instructions passed to GPT-OSS model" },
                    metadata: { type: ["object", "null"], description: "Additional metadata from GPT-OSS models" },
                    model: { type: "string", description: "Model name used for generation" },
                    object: { type: "string", enum: ["response"], description: "Object type for GPT-OSS models" },
                    output: {
                      type: "array",
                      description: "Array of output objects from GPT-OSS models including reasoning and message",
                      items: {
                        type: "object",
                        properties: {
                          id: { type: "string" },
                          content: {
                            type: "array",
                            items: {
                              type: "object",
                              properties: {
                                text: { type: "string" },
                                type: { type: "string", enum: ["reasoning_text", "output_text"] },
                                annotations: { type: "array" },
                                logprobs: { type: ["object", "null"] }
                              }
                            }
                          },
                          summary: { type: "array" },
                          type: { type: "string", enum: ["reasoning", "message"] },
                          encrypted_content: { type: ["string", "null"] },
                          status: { type: ["string", "null"] },
                          role: { type: ["string", "null"], enum: ["assistant"] }
                        }
                      }
                    },
                    parallel_tool_calls: { type: "boolean" },
                    temperature: { type: "number" },
                    tool_choice: { type: "string" },
                    tools: { type: "array" },
                    top_p: { type: "number" },
                    background: { type: "boolean" },
                    max_output_tokens: { type: "number" },
                    max_tool_calls: { type: ["number", "null"] },
                    previous_response_id: { type: ["string", "null"] },
                    prompt: { type: ["string", "null"] },
                    reasoning: { type: ["object", "null"] },
                    service_tier: { type: "string" },
                    status: { type: "string", enum: ["completed"] },
                    text: { type: ["string", "null"] },
                    top_logprobs: { type: "number" },
                    truncation: { type: "string" },
                    usage: {
                      type: "object",
                      properties: {
                        input_tokens: { type: "number" },
                        output_tokens: { type: "number" },
                        total_tokens: { type: "number" }
                      }
                    },
                    user: { type: ["string", "null"] }
                  }
                }
              ]
            },
            metadata: {
              type: "object",
              properties: {
                timestamp: { type: "string" },
                requestId: { type: "string" },
                model: { type: "string" },
                processingTime: { type: "number" },
                tokensUsed: { type: "number" }
              }
            }
          }
        },
        examples: {
                    request: {
            system: "You are a helpful assistant that explains complex topics in simple terms.",
            user: "What is machine learning?",
            max_tokens: 150,
            temperature: 0.7
          },
          response: {
            success: true,
            data: {
              response: "Machine learning is a way for computers to learn and make decisions by looking at lots of examples, just like how you learn to recognize cats by seeing many different cats.",
              usage: {
                prompt_tokens: 42,
                completion_tokens: 28,
                total_tokens: 70
              }
            },
            metadata: {
              timestamp: "2024-01-01T00:00:00.000Z",
              requestId: "req_1704067200000_abc123",
              model: "@cf/meta/llama-3.1-8b-instruct",
              processingTime: 1500,
              tokensUsed: 28
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "system": "You are a helpful assistant that explains complex topics in simple terms.",
    "user": "What is machine learning?",
    "max_tokens": 150,
    "temperature": 0.7
  }'`
        },
      },
      {
        method: "POST",
        path: "/api/v1/generate-gpt-oss",
        summary: "GPT-OSS Text Generation (Advanced)",
        description: "GPT-OSS models (@cf/openai/gpt-oss-120b, @cf/openai/gpt-oss-20b) provide advanced reasoning capabilities with step-by-step thinking. These models include reasoning output, instructions parameter support, and enhanced problem-solving abilities.",
        authentication: true,
        requestSchema: {
          type: "object",
          required: ["input"],
          properties: {
            input: {
              type: "string",
              description: "User input/question for GPT-OSS model"
            },
            instructions: {
              type: "string",
              description: "System instructions for GPT-OSS model behavior"
            },
            max_tokens: {
              type: "number",
              minimum: 1,
              maximum: 130991,
              default: 512,
              description: "Maximum number of tokens to generate (GPT-OSS models support up to ~130K)"
            },
            reasoning: {
              type: "object",
              description: "Reasoning configuration for GPT-OSS models",
              properties: {
                effort: {
                  type: "string",
                  enum: ["low", "medium", "high"],
                  default: "medium",
                  description: "Reasoning effort level"
                }
              }
            }
          }
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                id: { type: "string", description: "Unique response identifier" },
                created_at: { type: "number", description: "Creation timestamp" },
                instructions: { type: ["string", "null"], description: "Instructions used" },
                model: { type: "string", description: "Model used" },
                object: { type: "string", enum: ["response"] },
                output: {
                  type: "array",
                  description: "Response outputs including reasoning",
                  items: {
                    type: "object",
                    properties: {
                      id: { type: "string" },
                      content: {
                        type: "array",
                        items: {
                          type: "object",
                          properties: {
                            text: { type: "string" },
                            type: { type: "string", enum: ["reasoning_text", "output_text"] }
                          }
                        }
                      },
                      type: { type: "string", enum: ["reasoning", "message"] }
                    }
                  }
                },
                usage: {
                  type: "object",
                  properties: {
                    input_tokens: { type: "number" },
                    output_tokens: { type: "number" },
                    total_tokens: { type: "number" }
                  }
                }
              }
            },
            metadata: { type: "object" }
          }
        },
        examples: {
          request: {
            model: "@cf/openai/gpt-oss-120b",
            input: "Solve step by step: If a train travels 300 miles in 4 hours, what is its average speed?",
            instructions: "You are a helpful math tutor that shows step-by-step solutions.",
            reasoning: { effort: "high" },
            max_tokens: 200
          },
          response: {
            success: true,
            data: {
              id: "resp_42a3c3ea67a14dddb95f55a5e7e99b3b",
              created_at: 1764001620,
              instructions: "You are a helpful math tutor that shows step-by-step solutions.",
              model: "@cf/openai/gpt-oss-120b",
              object: "response",
              output: [
                {
                  id: "rs_83dd1a02fbec43f4ab5136a53be2656d",
                  content: [{
                    text: "The user asks to solve a math problem step by step. I need to calculate average speed = distance/time = 300 miles / 4 hours = 75 mph. I should show each step clearly.",
                    type: "reasoning_text"
                  }],
                  type: "reasoning"
                },
                {
                  id: "msg_075035ec63714208bc8f016c440c2c53",
                  content: [{
                    text: "**Step 1:** Identify the given information: Distance = 300 miles, Time = 4 hours\n**Step 2:** Use the formula: Average Speed = Total Distance ÷ Total Time\n**Step 3:** Calculate: 300 ÷ 4 = 75 mph\n**Answer:** The train's average speed is **75 miles per hour**.",
                    type: "output_text"
                  }],
                  type: "message",
                  role: "assistant"
                }
              ],
              usage: {
                input_tokens: 23,
                output_tokens: 345,
                total_tokens: 368
              }
            },
            metadata: {
              timestamp: "2025-11-24T16:27:08.251Z",
              requestId: "req_1764001620745_hk2ruc8vk",
              model: "@cf/openai/gpt-oss-120b",
              processingTime: 7506
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/openai/gpt-oss-120b",
    "input": "Explain quantum computing in simple terms",
    "instructions": "You are a physics professor who makes complex topics accessible.",
    "reasoning": { "effort": "high" },
    "max_tokens": 500
  }'`
        }
      },
      {
        method: "POST",
        path: "/api/v1/embeddings",
        summary: "Text Embeddings",
        description: "Generates vector embeddings for text input. Useful for semantic search, similarity comparisons, and other NLP tasks.",
        authentication: true,
        requestSchema: {
          type: "object",
          required: ["input"],
          properties: {
            input: {
              oneOf: [
                { type: "string" },
                { type: "array", items: { type: "string" } }
              ],
              description: "Text or array of texts to embed"
            },
            model: {
              type: "string",
              default: "@cf/baai/bge-base-en-v1.5",
              description: "Embedding model to use"
            }
          }
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                data: {
                  type: "array",
                  items: {
                    type: "array",
                    items: { type: "number" }
                  }
                },
                shape: { type: "array", items: { type: "number" } },
                usage: { type: "object" }
              }
            },
            metadata: { type: "object" }
          }
        },
        examples: {
          request: {
            input: "Artificial intelligence is transforming the world",
            model: "@cf/baai/bge-base-en-v1.5"
          },
          response: {
            success: true,
            data: {
              data: [[0.123, -0.456, 0.789, "... (768 dimensions)"]],
              shape: [1, 768],
              usage: { prompt_tokens: 9, total_tokens: 9 }
            },
            metadata: {
              requestId: "req_1704067200000_def456",
              processingTime: 890
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/embeddings \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "input": "Artificial intelligence is transforming the world"
  }'`
        }
      },
      {
        method: "POST",
        path: "/api/v1/images",
        summary: "Image Generation",
        description: "Generates high-quality images from text prompts using Stable Diffusion models. Returns base64-encoded image data ready for saving as PNG/JPEG files.",
        authentication: true,
        requestSchema: {
          type: "object",
          required: ["prompt", "model"],
          properties: {
            model: { 
              type: "string", 
              enum: ["@cf/stabilityai/stable-diffusion-xl-base-1.0", "@cf/bytedance/stable-diffusion-xl-lightning"],
              description: "Image generation model: stable-diffusion-xl-base-1.0 (high quality, 4-6s) or xl-lightning (fast, 2-4s)"
            },
            prompt: { type: "string", description: "Detailed text description of the image to generate. Use descriptive terms like 'photorealistic', 'highly detailed', '4k quality'" },
            width: { 
              type: "number", 
              minimum: 256, 
              maximum: 2048, 
              default: 512,
              description: "Image width in pixels (higher values take longer but produce better quality)"
            },
            height: { 
              type: "number", 
              minimum: 256, 
              maximum: 2048, 
              default: 512,
              description: "Image height in pixels (higher values take longer but produce better quality)"
            },
            num_steps: { 
              type: "number", 
              minimum: 1, 
              maximum: 20, 
              default: 20,
              description: "Number of diffusion steps (higher = better quality but slower)"
            },
            guidance: { 
              type: "number", 
              minimum: 1, 
              maximum: 20, 
              default: 7.5,
              description: "How closely to follow the prompt (7.5-15 recommended for better adherence)"
            },
            negative_prompt: {
              type: "string",
              description: "Elements to avoid in the generated image (e.g., 'blurry, low quality, distorted')"
            },
            seed: {
              type: "number",
              description: "Random seed for reproducible results (same seed + prompt = same image)"
            }
          }
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                image: { type: "string", description: "Base64-encoded image data" },
                format: { type: "string", description: "Image format (png/jpeg)" },
                size: { type: "number", description: "Image file size in bytes" },
                contentType: { type: "string", description: "MIME content type" }
              }
            },
            metadata: { 
              type: "object",
              properties: {
                requestId: { type: "string" },
                model: { type: "string" },
                processingTime: { type: "number", description: "Generation time in milliseconds" }
              }
            }
          }
        },
        examples: {
          request: {
            model: "@cf/stabilityai/stable-diffusion-xl-base-1.0",
            prompt: "A majestic eagle soaring over snowy mountains, highly detailed, award-winning photography",
            width: 1024,
            height: 1024,
            guidance: 7.5,
            num_steps: 20
          },
          response: {
            success: true,
            data: {
              image: "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOt...", // Base64 image data
              format: "png",
              size: 306108,
              contentType: "image/png"
            },
            metadata: {
              requestId: "req_1759055109536_6sbo5kq0h",
              model: "@cf/stabilityai/stable-diffusion-xl-base-1.0",
              processingTime: 4839
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/images \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A beautiful sunset over a mountain landscape, photorealistic, 4k quality",
    "width": 1024,
    "height": 1024,
    "guidance": 7.5,
    "num_steps": 20
  }'`
        }
      },
      {
        method: "POST",
        path: "/api/v1/classify",
        summary: "Text Classification",
        description: "Classifies text into predefined categories or performs sentiment analysis.",
        authentication: true,
        requestSchema: {
          type: "object",
          required: ["text"],
          properties: {
            text: { type: "string", description: "Text to classify" },
            categories: { 
              type: "array", 
              items: { type: "string" },
              description: "Optional list of categories for classification"
            },
            model: { 
              type: "string", 
              default: "@cf/huggingface/distilbert-sst-2-int8",
              description: "Classification model to use"
            }
          }
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  label: { type: "string" },
                  score: { type: "number" }
                }
              }
            },
            metadata: { type: "object" }
          }
        },
        examples: {
          request: {
            text: "I love this new AI service! It's incredibly helpful.",
            categories: ["positive", "negative", "neutral"]
          },
          response: {
            success: true,
            data: [
              { label: "POSITIVE", score: 0.9987 },
              { label: "NEGATIVE", score: 0.0013 }
            ],
            metadata: {
              requestId: "req_1704067200000_jkl012",
              processingTime: 1200
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/classify \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "text": "I love this new AI service!",
    "categories": ["positive", "negative", "neutral"]
  }'`
        }
      },
      {
        method: "POST",
        path: "/api/v1/summarize",
        summary: "Text Summarization",
        description: "Summarizes long text into shorter, more concise versions while preserving key information. Supports both legacy text format and new system+user prompt format.",
        authentication: true,
        requestSchema: {
          type: "object",
          properties: {
                        system: { 
              type: "string", 
              description: "System prompt defining summarization style and approach" 
            },
            user: { 
              type: "string", 
              description: "Text to summarize with any specific instructions" 
            },
                        text: { 
              type: "string", 
              description: "Text to summarize (legacy format)" 
            },
            style: { 
              type: "string", 
              default: "concise",
              description: "Summarization style (concise, detailed, bullet-points, etc.)"
            },
            max_length: { 
              type: "number", 
              default: 256,
              description: "Maximum length of the summary in tokens"
            }
          },
          oneOf: [
            { required: ["system", "user"] },
            { required: ["user"] },
            { required: ["text"] }
          ]
        },
        responseSchema: {
          type: "object",
          properties: {
            success: { type: "boolean" },
            data: {
              type: "object",
              properties: {
                response: { type: "string" },
                usage: { type: "object" }
              }
            },
            metadata: { type: "object" }
          }
        },
        examples: {
          request: {
            system: "You are a professional summarizer who creates clear, concise summaries that capture the most important points.",
            user: "Summarize this research paper: Artificial intelligence has made significant advances in natural language processing, computer vision, and robotics. Recent developments include more efficient training methods, better model architectures, and improved performance on benchmark tasks.",
            max_length: 100
          },
          response: {
            success: true,
            data: {
              response: "AI has advanced significantly in NLP, computer vision, and robotics through improved training methods, model architectures, and benchmark performance.",
              usage: {
                prompt_tokens: 89,
                completion_tokens: 23,
                total_tokens: 112
              }
            },
            metadata: {
              requestId: "req_1704067200000_mno345",
              processingTime: 1800
            }
          },
          curl: `curl -X POST ${baseUrl}/api/v1/summarize \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "system": "You are a professional summarizer.",
    "user": "Summarize this: [long text here]",
    "max_length": 100
  }'`
        }
      }
    ],
    models: {
      "text-generation": {
        description: "Large language models for text generation, completion, and conversation. GPT-OSS models provide advanced reasoning capabilities.",
        models: [
          "@cf/openai/gpt-oss-120b",
          "@cf/openai/gpt-oss-20b",
          "@cf/meta/llama-4-scout-17b-16e-instruct",
          "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
          "@cf/meta/llama-3.1-8b-instruct-fast",
          "@cf/ibm/granite-4.0-h-micro",
          "@cf/gemma/gemma-sea-lion-v4-27b-it",
          "@cf/google/gemma-3-12b-it",
          "@cf/mistralai/mistral-small-3.1-24b-instruct",
          "@cf/qwen/qwq-32b",
          "@cf/qwen/qwen2.5-coder-32b-instruct",
          "@cf/meta/llama-guard-3-8b",
          "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
          "@cf/meta/llama-3.2-1b-instruct",
          "@cf/meta/llama-3.2-3b-instruct",
          "@cf/meta/llama-3.2-11b-vision-instruct",
          "@cf/meta/llama-3.1-8b-instruct-awq",
          "@cf/meta/llama-3.1-8b-instruct-fp8",
          "@cf/meta/llama-3.1-8b-instruct",
          "@cf/meta/meta-llama-3-8b-instruct",
          "@cf/meta/llama-3-8b-instruct-awq",
          "@cf/meta/llama-3-8b-instruct",
          "@cf/microsoft/phi-2",
          "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
          "@cf/qwen/qwen1.5-14b-chat-awq",
          "@cf/qwen/qwen1.5-7b-chat-awq",
          "@cf/qwen/qwen1.5-0.5b-chat",
          "@cf/thebloke/discolm-german-7b-v1-awq",
          "@cf/tiiuae/falcon-7b-instruct",
          "@cf/openchat/openchat-3.5-0106",
          "@cf/defog/sqlcoder-7b-2",
          "@cf/deepseek/deepseek-math-7b-instruct",
          "@cf/nexusflow/starling-lm-7b-beta",
          "@cf/nousresearch/hermes-2-pro-mistral-7b",
          "@cf/mistralai/mistral-7b-instruct-v0.2-lora",
          "@cf/qwen/qwen1.5-1.8b-chat"
        ],
        defaultModel: "@cf/meta/llama-3.1-8b-instruct",
        parameters: {
          max_tokens: { min: 1, max: 8192, default: 512, description: "Maximum tokens to generate (GPT-OSS models support up to ~130K)" },
          temperature: { min: 0, max: 2, default: 0.7, description: "Randomness control (0=deterministic)" },
          top_p: { min: 0, max: 1, default: 1, description: "Nucleus sampling parameter" },
          stream: { type: "boolean", default: false, description: "Whether to stream the response" },
          reasoning: {
            type: "object",
            description: "GPT-OSS specific reasoning parameters",
            properties: {
              effort: { type: "string", enum: ["low", "medium", "high"], description: "Reasoning effort level (GPT-OSS only)" }
            }
          }
        },
        "specialFeatures": {
          "gpt-oss": {
            "models": ["@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b"],
            "features": [
              "Advanced step-by-step reasoning",
              "Built-in reasoning output display",
              "Higher token limits (~130K)",
              "Instructions parameter support",
              "Enhanced problem-solving capabilities"
            ],
            "inputFormat": {
              "system": "→ instructions",
              "user": "→ input"
            },
            "outputFormat": "Structured with separate reasoning and response sections"
          }
        }
      },
      "text-embeddings": {
        description: "Models for converting text into vector embeddings for semantic search and similarity",
        models: [
          "@cf/pfnet/plamo-embedding-1b",
          "@cf/google/embeddinggemma-300m",
          "@cf/baai/bge-reranker-base",
          "@cf/baai/bge-m3",
          "@cf/baai/bge-large-en-v1.5",
          "@cf/baai/bge-small-en-v1.5",
          "@cf/baai/bge-base-en-v1.5"
        ],
        defaultModel: "@cf/baai/bge-base-en-v1.5"
      },
      "text-to-image": {
        description: "Stable Diffusion models for generating high-quality images from text prompts. Returns base64-encoded PNG/JPEG data ready for saving. stable-diffusion-xl-base-1.0: High quality, detailed images (4-6s). xl-lightning: Fast generation, good quality (2-4s).",
        models: [
          "@cf/leonardo/lucid-origin",
          "@cf/leonardo/phoenix-1.0",
          "@cf/blackforestlabs/flux-1-schnell",
          "@cf/bytedance/stable-diffusion-xl-lightning",
          "@cf/lykon/dreamshaper-8-lcm",
          "@cf/runwayml/stable-diffusion-v1-5-img2img",
          "@cf/runwayml/stable-diffusion-v1-5-inpainting",
          "@cf/stabilityai/stable-diffusion-xl-base-1.0"
        ],
        defaultModel: "@cf/bytedance/stable-diffusion-xl-lightning",
        parameters: {
          num_steps: { min: 1, max: 20, default: 20, description: "Number of diffusion steps (higher = better quality)" },
          guidance: { min: 1, max: 20, default: 7.5, description: "Prompt adherence strength (7.5-15 recommended)" },
          width: { min: 256, max: 2048, default: 512, description: "Image width in pixels" },
          height: { min: 256, max: 2048, default: 512, description: "Image height in pixels" },
          negative_prompt: { type: "string", description: "Elements to avoid (e.g., 'blurry, low quality')" },
          seed: { type: "number", description: "Random seed for reproducible results" }
        }
      },
      "text-classification": {
        description: "Models for categorizing text and sentiment analysis",
        models: [
          "@cf/baai/bge-reranker-base",
          "@cf/huggingface/distilbert-sst-2-int8",
          "@cf/meta/llama-guard-3-8b"
        ],
        defaultModel: "@cf/huggingface/distilbert-sst-2-int8"
      },
      "automatic-speech-recognition": {
        description: "Models for converting speech audio to text",
        models: [
          "@cf/deepgram/flux",
          "@cf/deepgram/nova-3",
          "@cf/openai/whisper-large-v3-turbo",
          "@cf/openai/whisper-tiny-en",
          "@cf/openai/whisper"
        ],
        defaultModel: "@cf/openai/whisper"
      }
    },
    errorCodes: [
      {
        code: 400,
        description: "Bad Request - Invalid input, missing required fields, or parameter validation failed",
        example: {
          success: false,
          error: "Parameter validation failed: max_tokens must be between 1 and 8192",
          metadata: {
            timestamp: "2024-01-01T00:00:00.000Z",
            requestId: "req_1704067200000_error1"
          }
        }
      },
      {
        code: 401,
        description: "Unauthorized - Missing or invalid API key",
        example: {
          success: false,
          error: "Invalid API key",
          metadata: {
            timestamp: "2024-01-01T00:00:00.000Z",
            requestId: "req_1704067200000_error2"
          }
        }
      },
      {
        code: 404,
        description: "Not Found - Endpoint does not exist",
        example: {
          success: false,
          error: "Endpoint not found",
          metadata: {
            timestamp: "2024-01-01T00:00:00.000Z",
            requestId: "req_1704067200000_error3"
          }
        }
      },
      {
        code: 500,
        description: "Internal Server Error - Unexpected error occurred",
        example: {
          success: false,
          error: "Internal server error",
          metadata: {
            timestamp: "2024-01-01T00:00:00.000Z",
            requestId: "req_1704067200000_error4"
          }
        }
      },
      {
        code: 503,
        description: "Service Unavailable - AI service is temporarily unavailable",
        example: {
          success: false,
          error: "AI service unavailable",
          metadata: {
            timestamp: "2024-01-01T00:00:00.000Z",
            requestId: "req_1704067200000_error5"
          }
        }
      }
    ]
  }
}

export function generateHTMLDocumentation(docs: APIDocumentation): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${docs.info.title} - API Documentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; border-radius: 12px; }
        h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .subtitle { font-size: 1.2rem; opacity: 0.9; }
        .version { background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.9rem; display: inline-block; margin-top: 1rem; }
        .section { background: white; padding: 2rem; margin-bottom: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .section h2 { color: #4a5568; font-size: 1.8rem; margin-bottom: 1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
        .section h3 { color: #2d3748; font-size: 1.3rem; margin: 1.5rem 0 1rem; }
        .endpoint { border: 1px solid #e2e8f0; border-radius: 8px; margin: 1rem 0; overflow: hidden; }
        .endpoint-header { padding: 1rem; background: #f7fafc; display: flex; align-items: center; gap: 1rem; }
        .method { padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: bold; font-size: 0.85rem; color: white; }
        .method.GET { background: #38a169; }
        .method.POST { background: #3182ce; }
        .path { font-family: 'Monaco', 'Menlo', monospace; font-weight: bold; color: #2d3748; }
        .auth-required { background: #fed7d7; color: #c53030; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .auth-not-required { background: #c6f6d5; color: #38a169; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .endpoint-content { padding: 1rem; }
        .code-block { background: #1a202c; color: #e2e8f0; padding: 1rem; border-radius: 6px; overflow-x: auto; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9rem; margin: 1rem 0; }
        .json { background: #f7fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 6px; font-family: monospace; white-space: pre-wrap; font-size: 0.9rem; margin: 0.5rem 0; }
        .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .model-card { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; }
        .model-card h4 { color: #4a5568; margin-bottom: 0.5rem; }
        .model-list { list-style: none; margin: 0.5rem 0; }
        .model-list li { background: white; margin: 0.25rem 0; padding: 0.5rem; border-radius: 4px; border-left: 3px solid #667eea; font-family: monospace; font-size: 0.85rem; }
        .error-codes { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
        .error-card { border: 1px solid #fed7d7; background: #fffaf0; border-radius: 8px; padding: 1rem; }
        .error-code { font-size: 1.1rem; font-weight: bold; color: #c53030; }
        .tabs { display: flex; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem; }
        .tab { padding: 0.5rem 1rem; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab.active { border-bottom-color: #667eea; color: #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .highlight { background: #fef5e7; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #f6ad55; margin: 1rem 0; }
        .toc { background: #f7fafc; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }
        .toc ul { list-style: none; }
        .toc a { color: #4a5568; text-decoration: none; }
        .toc a:hover { color: #667eea; }
        @media (max-width: 768px) {
            .container { padding: 10px; }
            h1 { font-size: 2rem; }
            .model-grid, .error-codes { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>${docs.info.title}</h1>
            <div class="subtitle">${docs.info.description}</div>
            <div class="version">Version ${docs.info.version}</div>
        </header>

        <div class="toc section">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#authentication">Authentication</a></li>
                <li><a href="#endpoints">API Endpoints</a></li>
                <li><a href="#models">Available Models</a></li>
                <li><a href="#errors">Error Codes</a></li>
                <li><a href="#examples">Usage Examples</a></li>
            </ul>
        </div>

        <div class="section" id="authentication">
            <h2>🔐 Authentication</h2>
            <p><strong>Type:</strong> ${docs.authentication.type}</p>
            <p>${docs.authentication.description}</p>
            <div class="highlight">
                <strong>Header:</strong> <code>${docs.authentication.headerName}: ${docs.authentication.example}</code>
            </div>
        </div>

        <div class="section" id="endpoints">
            <h2>🚀 API Endpoints</h2>
            ${docs.endpoints.map(endpoint => `
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method ${endpoint.method}">${endpoint.method}</span>
                        <span class="path">${endpoint.path}</span>
                        <span class="${endpoint.authentication ? 'auth-required' : 'auth-not-required'}">
                            ${endpoint.authentication ? '🔒 Auth Required' : '🌐 Public'}
                        </span>
                    </div>
                    <div class="endpoint-content">
                        <h3>${endpoint.summary}</h3>
                        <p>${endpoint.description}</p>
                        
                        <div class="tabs">
                            <div class="tab active" onclick="showTab(event, 'example-${endpoint.path.replace(/[^a-zA-Z]/g, '')}')">Example</div>
                            ${endpoint.requestSchema ? '<div class="tab" onclick="showTab(event, \'request-' + endpoint.path.replace(/[^a-zA-Z]/g, '') + '\')">Request Schema</div>' : ''}
                            <div class="tab" onclick="showTab(event, 'response-${endpoint.path.replace(/[^a-zA-Z]/g, '')}')">Response Schema</div>
                        </div>
                        
                        <div id="example-${endpoint.path.replace(/[^a-zA-Z]/g, '')}" class="tab-content active">
                            <h4>cURL Example:</h4>
                            <div class="code-block">${endpoint.examples.curl}</div>
                            ${endpoint.examples.request ? `
                                <h4>Request Body:</h4>
                                <div class="json">${JSON.stringify(endpoint.examples.request, null, 2)}</div>
                            ` : ''}
                            <h4>Response:</h4>
                            <div class="json">${JSON.stringify(endpoint.examples.response, null, 2)}</div>
                        </div>
                        
                        ${endpoint.requestSchema ? `
                            <div id="request-${endpoint.path.replace(/[^a-zA-Z]/g, '')}" class="tab-content">
                                <div class="json">${JSON.stringify(endpoint.requestSchema, null, 2)}</div>
                            </div>
                        ` : ''}
                        
                        <div id="response-${endpoint.path.replace(/[^a-zA-Z]/g, '')}" class="tab-content">
                            <div class="json">${JSON.stringify(endpoint.responseSchema, null, 2)}</div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>

        <div class="section" id="models">
            <h2>🤖 Available Models</h2>
            <div class="model-grid">
                ${Object.entries(docs.models).map(([category, info]) => `
                    <div class="model-card">
                        <h4>${category.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
                        <p>${info.description}</p>
                        <p><strong>Default:</strong> <code>${info.defaultModel}</code></p>
                        <ul class="model-list">
                            ${info.models.map(model => `<li>${model}</li>`).join('')}
                        </ul>
                        ${info.parameters ? `
                            <details>
                                <summary>Parameters</summary>
                                <div class="json">${JSON.stringify(info.parameters, null, 2)}</div>
                            </details>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="section" id="errors">
            <h2>⚠️ Error Codes</h2>
            <div class="error-codes">
                ${docs.errorCodes.map(error => `
                    <div class="error-card">
                        <div class="error-code">${error.code}</div>
                        <p>${error.description}</p>
                        <div class="json">${JSON.stringify(error.example, null, 2)}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="section" id="examples">
            <h2>💡 Usage Examples</h2>
            
            <h3>🎨 Image Generation</h3>
            <p>Generate high-quality images from text prompts. The API returns base64-encoded image data ready for saving:</p>
            <div class="code-block">curl -X POST ${docs.servers[0].url}/api/v1/images \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A majestic eagle soaring over snowy mountains, highly detailed, award-winning photography",
    "width": 1024,
    "height": 1024,
    "guidance": 7.5,
    "num_steps": 20
  }'</div>

            <h4>📁 Save Generated Image (Python)</h4>
            <div class="code-block">import requests
import base64
import json

response = requests.post('${docs.servers[0].url}/api/v1/images', 
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
    
    print(f"Image saved! Size: {data['size']} bytes")</div>

            <h3>🧠 GPT-OSS Advanced Reasoning Models</h3>
            <p>GPT-OSS models provide enhanced reasoning capabilities with step-by-step thinking:</p>
            <div class="code-block">curl -X POST ${docs.servers[0].url}/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/openai/gpt-oss-120b",
    "system": "You are a helpful math tutor that shows step-by-step solutions.",
    "user": "Solve step by step: If a train travels 300 miles in 4 hours, what is its average speed?",
    "reasoning": { "effort": "high" },
    "max_tokens": 200
  }'</div>

            <h4>🔬 GPT-OSS Response Format</h4>
            <p>GPT-OSS models return structured responses with separate reasoning and answer sections:</p>
            <div class="json">{
  "success": true,
  "data": {
    "id": "resp_42a3c3ea67a14dddb95f55a5e7e99b3b",
    "model": "@cf/openai/gpt-oss-120b",
    "output": [
      {
        "type": "reasoning",
        "content": [
          {
            "type": "reasoning_text",
            "text": "I need to calculate average speed = distance/time = 300 miles / 4 hours = 75 mph..."
          }
        ]
      },
      {
        "type": "message",
        "role": "assistant",
        "content": [
          {
            "type": "output_text",
            "text": "**Step 1:** Distance = 300 miles, Time = 4 hours\\n**Step 2:** Formula: Speed = Distance ÷ Time\\n**Answer:** 75 mph"
          }
        ]
      }
    ],
    "usage": {
      "input_tokens": 23,
      "output_tokens": 345,
      "total_tokens": 368
    }
  }
}</div>

            <h3>💭 System + User Prompts (Enhanced Text Generation)</h3>
            <p>The API supports separate system and user prompts for enhanced control over AI behavior:</p>
            <div class="code-block">curl -X POST ${docs.servers[0].url}/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "system": "You are a helpful coding assistant. Provide clear, concise answers.",
    "user": "How do I create a REST API in Python?",
    "max_tokens": 500,
    "temperature": 0.7
  }'</div>

            <h3>🎭 Different AI Personalities</h3>
            <div class="code-block"># Professional consultant
{
  "system_prompt": "You are a business consultant providing structured analysis.",
  "user_prompt": "How to improve marketing strategy?"
}

# Creative poet
{
  "system_prompt": "You are a poet who writes only haikus.",
  "user_prompt": "Write about artificial intelligence"
}

# Casual friend
{
  "system_prompt": "You're a friendly AI buddy. Use emojis and casual language.",
  "user_prompt": "Explain machine learning"
}</div>

            <h3>🔄 Backward Compatibility</h3>
            <p>Legacy single prompt format is still supported:</p>
            <div class="code-block">{
  "model": "@cf/meta/llama-3.1-8b-instruct",
  "prompt": "Explain artificial intelligence in simple terms",
  "max_tokens": 100
}</div>

            <h3>🖼️ Image Generation Pro Tips</h3>
            <div class="highlight">
                <strong>Best Practices:</strong>
                <ul style="margin-top: 0.5rem; padding-left: 1rem;">
                    <li>Use descriptive prompts: "photorealistic", "highly detailed", "award-winning"</li>
                    <li>Include style keywords: "digital art", "oil painting", "photography"</li>
                    <li>Add quality terms: "4k", "8k", "ultra-detailed", "masterpiece"</li>
                    <li>Use negative_prompt to exclude: "blurry", "low quality", "distorted"</li>
                    <li>Higher resolutions (1024x1024) take longer but produce better results</li>
                    <li>stable-diffusion-xl-base-1.0: Best quality (4-6s), xl-lightning: Fast (2-4s)</li>
                </ul>
            </div>
        </div>

        <footer style="text-align: center; padding: 2rem; color: #718096;">
            <p>Built with ❤️ using Cloudflare Workers AI</p>
            <p>Last updated: ${new Date().toISOString()}</p>
        </footer>
    </div>

    <script>
        function showTab(evt, tabName) {
            var i, tabcontent, tabs;
            tabcontent = evt.target.closest('.endpoint-content').getElementsByClassName('tab-content');
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove('active');
            }
            tabs = evt.target.closest('.tabs').getElementsByClassName('tab');
            for (i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            document.getElementById(tabName).classList.add('active');
            evt.target.classList.add('active');
        }

                document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
  `
}

export function generateMarkdownDocumentation(docs: APIDocumentation): string {
  return `# ${docs.info.title}

${docs.info.description}

**Version:** ${docs.info.version}
**Contact:** [API Support](${docs.info.contact.url})

## 🔐 Authentication

**Type:** ${docs.authentication.type}

${docs.authentication.description}

**Header:** \`${docs.authentication.headerName}: ${docs.authentication.example}\`

## 🚀 API Endpoints

### Table of Contents

${docs.endpoints.map((endpoint, index) =>
  `${index + 1}. [${endpoint.method} ${endpoint.path}](#${endpoint.method.toLowerCase()}-${endpoint.path.replace(/\//g, '-')}) - ${endpoint.summary}`
).join('\n')}

${docs.endpoints.map(endpoint => `
### ${endpoint.method} ${endpoint.path}

${endpoint.description}

**Authentication:** ${endpoint.authentication ? '🔒 Required' : '🌐 Public'}

**Request Example:**
\`\`\`bash
${endpoint.examples.curl}
\`\`\`

**Request Body:**
\`\`\`json
${JSON.stringify(endpoint.examples.request || {}, null, 2)}
\`\`\`

**Response:**
\`\`\`json
${JSON.stringify(endpoint.examples.response, null, 2)}
\`\`\`
`).join('\n---\n')}

## 🤖 Available Models

${Object.entries(docs.models).map(([category, info]) => `
### ${category.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}

${info.description}

**Default Model:** \`${info.defaultModel}\`

**Available Models:**
${info.models.map(model => `- \`${model}\``).join('\n')}

${info.parameters ? `
**Parameters:**
${Object.entries(info.parameters).map(([param, config]) =>
  `- \`${param}\`: ${config.description || ''} (min: ${config.min || 'N/A'}, max: ${config.max || 'N/A'}, default: ${config.default || 'N/A'})`
).join('\n')}
` : ''}
`).join('\n')}

## ⚠️ Error Codes

${docs.errorCodes.map(error => `
### ${error.code}

${error.description}

**Example:**
\`\`\`json
${JSON.stringify(error.example, null, 2)}
\`\`\`
`).join('\n')}

## 💡 Usage Examples

### 🎨 Image Generation

Generate high-quality images from text prompts:

\`\`\`bash
curl -X POST ${docs.servers[0].url}/api/v1/images \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A beautiful sunset over ocean waves, vibrant colors, peaceful",
    "width": 1024,
    "height": 1024,
    "guidance": 7.5,
    "num_steps": 20
  }'
\`\`\`

### 💭 System + User Prompts

The API supports separate system and user prompts for enhanced control:

\`\`\`bash
curl -X POST ${docs.servers[0].url}/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "system": "You are a helpful coding assistant. Provide clear, concise answers.",
    "user": "How do I create a REST API in Python?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
\`\`\`

### 🖼️ Save Generated Image (Python)

\`\`\`python
import requests
import base64

response = requests.post('${docs.servers[0].url}/api/v1/images',
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
\`\`\`

## 📊 Response Format

All API responses follow this consistent structure:

\`\`\`json
{
  "success": true,
  "data": {
      },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "processingTime": 1250,
    "tokensUsed": 42
  }
}
\`\`\`

Error responses:
\`\`\`json
{
  "success": false,
  "error": "Detailed error message",
  "metadata": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "requestId": "req_1704067200000_abc123",
    "processingTime": 50
  }
}
\`\`\`

## 🔍 Monitoring & Debugging

### Request Tracking
Every request gets a unique ID for debugging across logs and responses.

### Performance Metrics
- Processing time measurement for each request
- Token usage tracking (when available)
- Model performance comparison

### Structured Logging
\`\`\`json
{
  "requestId": "req_1704067200000_abc123",
  "method": "POST",
  "path": "/api/v1/generate",
  "duration": 1250,
  "timestamp": "2024-01-01T00:00:00.000Z"
}
\`\`\`

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

\`\`\`bash
# Run type checking
npm run typecheck

# Test API endpoints locally
curl -X GET http://localhost:8787/api/v1/health

# Test with authentication
curl -X POST http://localhost:8787/api/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key" \\
  -d '{"prompt": "Hello, world!"}'
\`\`\`

---

**Built with ❤️ using Cloudflare Workers AI**
*Last updated: ${new Date().toISOString()}*
`
}