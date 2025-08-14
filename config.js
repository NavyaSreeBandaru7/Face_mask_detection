// In config.js file, find the OpenAI section and update:

const CONFIG = {
  // ... other config ...
  
  model: {
    // OpenAI API configuration
    openai: {
      // ADD YOUR OPENAI API KEY HERE
      apiKey: 'sk-your-actual-openai-api-key-here',
      model: 'gpt-4-vision-preview',
      maxTokens: 300,
      temperature: 0.1,
      
      // Enable OpenAI enhanced analysis
      enabled: true,  // Set to true to enable
      
      // Rate limiting
      requestsPerMinute: 20,
    },
    
    // ... other model config ...
  }
};

// Example with real key format:
// apiKey: 'sk-1234567890abcdef1234567890abcdef1234567890abcdef',
