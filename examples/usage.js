#!/usr/bin/env node
/**
 * Example usage of the Enhanced Cloudflare Workers AI API
 * Run with: node examples/usage.js
 */

const API_BASE_URL = 'http://localhost:8787/api/v1'
const API_KEY = 'your-api-key-here' // Replace with your actual API key

// Example requests for different AI tasks

// 1. Health Check
async function checkHealth() {
  console.log('🏥 Checking API health...')
  
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    const result = await response.json()
    
    console.log('✅ Health check:', result.success ? 'HEALTHY' : 'UNHEALTHY')
    console.log('📊 Response:', JSON.stringify(result, null, 2))
  } catch (error) {
    console.error('❌ Health check failed:', error.message)
  }
}

// 2. List Available Models
async function listModels() {
  console.log('\n📋 Listing available models...')
  
  try {
    const response = await fetch(`${API_BASE_URL}/models`)
    const result = await response.json()
    
    if (result.success) {
      console.log(`✅ Found ${result.data.total} models across ${Object.keys(result.data.models).length} categories`)
      
      for (const [category, config] of Object.entries(result.data.models)) {
        console.log(`\n🔧 ${category}:`)
        console.log(`   Default: ${config.defaultModel}`)
        console.log(`   Models: ${config.models.length}`)
        config.models.forEach(model => console.log(`     - ${model}`))
      }
    }
  } catch (error) {
    console.error('❌ Failed to list models:', error.message)
  }
}

// 3. Text Generation Example
async function generateText() {
  console.log('\n✍️ Generating text...')
  
  const requestBody = {
    prompt: "Write a short poem about artificial intelligence in 2024",
    max_tokens: 150,
    temperature: 0.8
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(requestBody)
    })
    
    const result = await response.json()
    
    if (result.success) {
      console.log('✅ Text generation successful')
      console.log(`⏱️  Processing time: ${result.metadata.processingTime}ms`)
      console.log(`🤖 Model: ${result.metadata.model}`)
      console.log(`📝 Generated text:\n${result.data.response || result.data.text || result.data}`)
    } else {
      console.error('❌ Text generation failed:', result.error)
    }
  } catch (error) {
    console.error('❌ Text generation error:', error.message)
  }
}

// 4. Text Embedding Example
async function generateEmbedding() {
  console.log('\n🧮 Generating text embedding...')
  
  const requestBody = {
    input: "Cloudflare Workers AI provides powerful machine learning capabilities"
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(requestBody)
    })
    
    const result = await response.json()
    
    if (result.success) {
      console.log('✅ Embedding generation successful')
      console.log(`⏱️  Processing time: ${result.metadata.processingTime}ms`)
      console.log(`🤖 Model: ${result.metadata.model}`)
      
      const embeddings = result.data.data || result.data.result || result.data
      if (Array.isArray(embeddings) && embeddings.length > 0) {
        console.log(`📊 Embedding dimensions: ${embeddings[0].length || 'N/A'}`)
        console.log(`🔢 First 5 values: [${embeddings[0].slice(0, 5).map(n => n.toFixed(4)).join(', ')}...]`)
      } else {
        console.log('📊 Embedding result:', typeof embeddings, Object.keys(embeddings || {}))
      }
    } else {
      console.error('❌ Embedding generation failed:', result.error)
    }
  } catch (error) {
    console.error('❌ Embedding generation error:', error.message)
  }
}

// 5. Text Classification Example
async function classifyText() {
  console.log('\n🏷️  Classifying text...')
  
  const requestBody = {
    text: "This new AI service is absolutely amazing! I love how fast and accurate it is.",
    categories: ["positive", "negative", "neutral"]
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(requestBody)
    })
    
    const result = await response.json()
    
    if (result.success) {
      console.log('✅ Text classification successful')
      console.log(`⏱️  Processing time: ${result.metadata.processingTime}ms`)
      console.log(`🤖 Model: ${result.metadata.model}`)
      console.log('🏷️  Classification results:', JSON.stringify(result.data, null, 2))
    } else {
      console.error('❌ Text classification failed:', result.error)
    }
  } catch (error) {
    console.error('❌ Text classification error:', error.message)
  }
}

// 6. Text Summarization Example
async function summarizeText() {
  console.log('\n📄 Summarizing text...')
  
  const longText = `
    Artificial Intelligence (AI) has rapidly evolved from a concept in science fiction to a transformative technology reshaping industries worldwide. In 2024, AI systems are more capable than ever, with large language models like GPT-4 and Claude demonstrating remarkable abilities in text generation, reasoning, and creative tasks.
    
    The development of AI has been marked by significant breakthroughs in machine learning, particularly deep learning and neural networks. These advances have enabled AI systems to process and understand human language, generate images from text descriptions, and even write code.
    
    However, with these advances come important considerations around ethics, safety, and the societal impact of AI. As AI becomes more integrated into our daily lives, from virtual assistants to autonomous vehicles, ensuring these systems are reliable, fair, and beneficial to humanity remains a critical challenge.
    
    Looking forward, the future of AI promises even more exciting developments, including more efficient models, better human-AI collaboration, and potentially artificial general intelligence (AGI) that could revolutionize how we work and live.
  `
  
  const requestBody = {
    text: longText.trim(),
    style: "concise",
    max_length: 100
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/summarize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(requestBody)
    })
    
    const result = await response.json()
    
    if (result.success) {
      console.log('✅ Text summarization successful')
      console.log(`⏱️  Processing time: ${result.metadata.processingTime}ms`)
      console.log(`🤖 Model: ${result.metadata.model}`)
      console.log(`📝 Summary:\n${result.data.response || result.data.text || result.data}`)
    } else {
      console.error('❌ Text summarization failed:', result.error)
    }
  } catch (error) {
    console.error('❌ Text summarization error:', error.message)
  }
}

// Main execution
async function main() {
  console.log('🚀 Enhanced Cloudflare Workers AI API - Usage Examples\n')
  console.log('=' .repeat(60))
  
  // Run examples sequentially
  await checkHealth()
  await listModels()
  
  // Skip authenticated requests if no API key is set
  if (API_KEY === 'your-api-key-here') {
    console.log('\n⚠️  Skipping authenticated requests - please set your API_KEY in the script')
    console.log('📝 To test with authentication:')
    console.log('   1. Start the dev server: npm run dev')
    console.log('   2. Set your API key in this script')
    console.log('   3. Run: node examples/usage.js')
  } else {
    await generateText()
    await generateEmbedding()
    await classifyText()
    await summarizeText()
  }
  
  console.log('\n' + '=' .repeat(60))
  console.log('✅ Examples completed!')
}

// Error handling
process.on('unhandledRejection', (error) => {
  console.error('❌ Unhandled promise rejection:', error)
  process.exit(1)
})

// Run the examples
main().catch(error => {
  console.error('❌ Main execution failed:', error)
  process.exit(1)
})