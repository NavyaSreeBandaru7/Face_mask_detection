/**
 * Configuration file for Face Mask Detection System
 * Customize these settings based on your requirements
 */

const CONFIG = {
  // Application settings
  app: {
    name: 'Face Mask Detection System',
    version: '1.0.0',
    description: 'Real-time AI-powered mask detection for public health compliance',
    author: 'Your Name',
    debugMode: false, // Set to true for development
  },

  // Camera settings
  camera: {
    defaultConstraints: {
      video: {
        width: { ideal: 1280, min: 640 },
        height: { ideal: 720, min: 480 },
        frameRate: { ideal: 30, min: 15 },
        facingMode: 'user', // 'user' for front camera, 'environment' for back
      },
      audio: false,
    },
    
    // Fallback constraints for older devices
    fallbackConstraints: {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 15 },
      },
      audio: false,
    },
  },

  // Detection settings
  detection: {
    // Default confidence threshold (0.1 to 1.0)
    defaultConfidence: 0.5,
    
    // Detection modes
    modes: {
      realtime: 100,    // Detection interval in ms for real-time mode
      interval: 5000,   // Detection interval in ms for interval mode
    },
    
    // Maximum number of faces to detect simultaneously
    maxDetections: 5,
    
    // Minimum face size (percentage of canvas width)
    minFaceSize: 0.05,
    
    // Face detection sensitivity
    faceDetectionThreshold: 0.3,
    
    // Mask classification confidence
    maskConfidence: {
      high: 0.8,    // High confidence threshold
      medium: 0.6,  // Medium confidence threshold
      low: 0.4,     // Low confidence threshold
    },
  },

  // AI/ML Model settings
  model: {
    // TensorFlow.js model configuration
    tensorflow: {
      backend: 'webgl', // 'webgl', 'cpu', or 'wasm'
      modelUrl: './models/mask_detection.json',
      weightsUrl: './models/mask_detection_weights.bin',
      inputSize: 224, // Model input image size
      
      // Model loading timeout (ms)
      loadTimeout: 30000,
      
      // Batch size for inference
      batchSize: 1,
    },
    
    // OpenAI API configuration (optional)
    openai: {
      apiKey: process.env.OPENAI_API_KEY || '', // Set in environment variables
      model: 'gpt-4-vision-preview',
      maxTokens: 300,
      temperature: 0.1,
      
      // Enable OpenAI enhanced analysis
      enabled: false,
      
      // Rate limiting
      requestsPerMinute: 20,
    },
    
    // Alternative detection method when AI is unavailable
    fallbackDetection: {
      enabled: true,
      method: 'pixel_analysis', // 'pixel_analysis' or 'pattern_matching'
      sensitivity: 0.6,
    },
  },

  // User Interface settings
  ui: {
    // Theme settings
    theme: {
      primary: '#667eea',
      secondary: '#764ba2',
      success: '#4CAF50',
      warning: '#ffc107',
      danger: '#f44336',
      info: '#2196F3',
    },
    
    // Animation settings
    animations: {
      enabled: true,
      duration: 300, // Animation duration in ms
      easing: 'ease-in-out',
    },
    
    // Responsive breakpoints
    breakpoints: {
      mobile: 768,
      tablet: 1024,
      desktop: 1200,
    },
    
    // Detection overlay settings
    overlay: {
      boundingBoxWidth: 3,
      labelFontSize: 12,
      labelPadding: 5,
      
      // Colors for different states
      colors: {
        mask: '#4CAF50',
        noMask: '#f44336',
        uncertain: '#ffc107',
      },
    },
  },

  // Audio settings
  audio: {
    // Alert sound settings
    alerts: {
      enabled: true,
      volume: 0.1, // 0.0 to 1.0
      
      // Sound types
      sounds: {
        violation: {
          frequency: 800,
          duration: 200,
          type: 'sine',
        },
        detection: {
          frequency: 600,
          duration: 100,
          type: 'square',
        },
        error: {
          frequency: 400,
          duration: 300,
          type: 'sawtooth',
        },
      },
    },
    
    // Voice announcements (if supported)
    voice: {
      enabled: false,
      language: 'en-US',
      rate: 1.0,
      pitch: 1.0,
      volume: 0.8,
    },
  },

  // Statistics and logging
  analytics: {
    // Enable statistics collection
    enabled: true,
    
    // Maximum log entries to keep
    maxLogEntries: 1000,
    
    // Statistics update interval (ms)
    updateInterval: 1000,
    
    // Export settings
    export: {
      formats: ['json', 'csv'],
      includeTimestamps: true,
      includeScreenshots: false,
    },
    
    // Performance monitoring
    performance: {
      trackFPS: true,
      trackMemory: true,
      trackDetectionTime: true,
    },
  },

  // Privacy and security settings
  privacy: {
    // Data retention
    dataRetention: {
      logs: 7, // Days to keep logs
      statistics: 30, // Days to keep statistics
      screenshots: 1, // Days to keep screenshots
    },
    
    // Privacy mode (additional restrictions)
    privacyMode: false,
    
    // Consent settings
    requireConsent: true,
    consentMessage: 'This application uses your camera for mask detection. No data is stored or transmitted.',
  },

  // Performance optimization
  performance: {
    // Canvas optimization
    canvas: {
      willReadFrequently: true,
      alpha: false,
      desynchronized: true,
    },
    
    // Memory management
    memory: {
      maxCanvasSize: 1920 * 1080,
      garbageCollectionInterval: 60000, // ms
      clearCanvasOnStop: true,
    },
    
    // Detection optimization
    detection: {
      skipFrames: 0, // Skip N frames between detections
      downscaleForDetection: 0.5, // Scale factor for detection processing
      useWorkers: false, // Use web workers for processing (experimental)
    },
  },

  // Development and debugging
  debug: {
    // Console logging levels
    logLevel: 'info', // 'debug', 'info', 'warn', 'error'
    
    // Visual debugging
    showFPS: false,
    showDetectionTime: false,
    showMemoryUsage: false,
    
    // Testing modes
    mockCamera: false,
    simulateDetections: false,
    
    // Performance profiling
    enableProfiling: false,
  },

  // Experimental features
  experimental: {
    // Multi-camera support
    multiCamera: false,
    
    // Advanced face tracking
    faceTracking: false,
    
    // Gesture detection
    gestureDetection: false,
    
    // Cloud model integration
    cloudModels: false,
    
    // Real-time streaming
    streaming: false,
  },
};

// Export configuration for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CONFIG;
} else if (typeof window !== 'undefined') {
  window.CONFIG = CONFIG;
}

// Validation function to check configuration integrity
function validateConfig(config) {
  const errors = [];
  
  // Check required fields
  if (!config.app.name) errors.push('App name is required');
  if (!config.app.version) errors.push('App version is required');
  
  // Validate ranges
  if (config.detection.defaultConfidence < 0.1 || config.detection.defaultConfidence > 1.0) {
    errors.push('Detection confidence must be between 0.1 and 1.0');
  }
  
  if (config.audio.alerts.volume < 0.0 || config.audio.alerts.volume > 1.0) {
    errors.push('Audio volume must be between 0.0 and 1.0');
  }
  
  // Check camera constraints
  if (config.camera.defaultConstraints.video.width.ideal < 320) {
    errors.push('Camera width should be at least 320px');
  }
  
  if (config.camera.defaultConstraints.video.height.ideal < 240) {
    errors.push('Camera height should be at least 240px');
  }
  
  return {
    isValid: errors.length === 0,
    errors: errors
  };
}

// Auto-validate configuration in debug mode
if (CONFIG.debug.logLevel === 'debug') {
  const validation = validateConfig(CONFIG);
  if (!validation.isValid) {
    console.warn('Configuration validation errors:', validation.errors);
  }
}
