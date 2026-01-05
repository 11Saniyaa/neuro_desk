import React, { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import { analyzeFrame } from './utils/api'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [darkMode, setDarkMode] = useState(() => {
    // Load from localStorage or default to false (with error handling for private browsing)
    try {
      const saved = localStorage.getItem('darkMode')
      return saved ? JSON.parse(saved) : false
    } catch (error) {
      console.warn('Failed to read from localStorage:', error)
      return false // Default to light mode if localStorage is unavailable
    }
  })
  const [showSettings, setShowSettings] = useState(false)
  const [sessionStartTime, setSessionStartTime] = useState(null)
  const [sessionDuration, setSessionDuration] = useState(0)
  
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const animationFrameRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const isConnectedRef = useRef(false) // Use ref to avoid stale closures
  const connectWebSocketRef = useRef(null) // Ref to store connectWebSocket function
  const abortControllersRef = useRef([]) // Track active AbortControllers for cleanup

  useEffect(() => {
    isConnectedRef.current = isConnected
  }, [isConnected])

  // Cleanup AbortControllers on unmount
  useEffect(() => {
    return () => {
      // Abort all active requests when component unmounts
      abortControllersRef.current.forEach(controller => {
        try {
          controller.abort()
        } catch (error) {
          // Ignore errors during cleanup
        }
      })
      abortControllersRef.current = []
    }
  }, [])

  // Update dark mode class on body
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
    // Save to localStorage with error handling (fails in private browsing)
    try {
      localStorage.setItem('darkMode', JSON.stringify(darkMode))
    } catch (error) {
      console.warn('Failed to save to localStorage:', error)
      // Continue without saving - preference will reset on reload
    }
  }, [darkMode])

  // Session timer
  useEffect(() => {
    let interval = null
    if (isConnected && sessionStartTime) {
      interval = setInterval(() => {
        setSessionDuration(Math.floor((Date.now() - sessionStartTime) / 1000))
      }, 1000)
    } else if (!isConnected) {
      setSessionDuration(0)
      setSessionStartTime(null)
    }
    
    // Cleanup: Always clear interval on unmount or dependency change
    return () => {
      if (interval) {
        clearInterval(interval)
        interval = null
      }
    }
  }, [isConnected, sessionStartTime])

  // Format session duration
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Space to start/stop (when not in input)
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault()
        if (isConnected) {
          stopCamera()
        } else {
          startCamera()
        }
      }
      // Escape to close settings
      if (e.code === 'Escape' && showSettings) {
        setShowSettings(false)
      }
      // 'E' to export data
      if (e.code === 'KeyE' && e.target.tagName !== 'INPUT' && data && data.productivity && !data.productivity.error) {
        exportData()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [isConnected, showSettings, data, stopCamera, exportData, startCamera])

  // Export data function
  const exportData = useCallback(() => {
    if (!data) return
    
    const exportData = {
      timestamp: new Date().toISOString(),
      session_duration: formatDuration(sessionDuration),
      productivity: data.productivity,
      posture: data.posture,
      eye_strain: data.eye_strain,
      engagement: data.engagement,
      stress: data.stress,
      recommendations: data.recommendations
    }

    // Create CSV
    const csvRows = []
    csvRows.push(['Metric', 'Value', 'Status'])
    csvRows.push(['Productivity Score', data.productivity?.productivity_score || 'N/A', ''])
    csvRows.push(['Posture Score', data.posture?.score || 'N/A', data.posture?.slouching ? 'Slouching' : 'Good'])
    csvRows.push(['Eye Strain Score', data.eye_strain?.score || 'N/A', data.eye_strain?.eye_strain_risk || 'N/A'])
    csvRows.push(['Engagement Score', data.engagement?.score || 'N/A', data.engagement?.concentration || 'N/A'])
    csvRows.push(['Stress Score', data.stress?.score || 'N/A', data.stress?.stress_level || 'N/A'])
    
    const csv = csvRows.map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `neuro_desk_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [data, sessionDuration])

  // Debug: Log when data changes
  useEffect(() => {
    if (data) {
      console.log('📊 Data state updated:', {
        hasProductivity: !!data.productivity,
        hasPosture: !!data.posture,
        hasEyeStrain: !!data.eye_strain,
        hasEngagement: !!data.engagement,
        hasStress: !!data.stress,
        hasRecommendations: !!data.recommendations,
        productivityScore: data.productivity?.productivity_score
      })
    }
  }, [data])

  useEffect(() => {
    return () => {
      // Cleanup
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  // Function to stop sending frames
  const stopSendingFrames = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }, [])

  // Function to send frames using HTTP POST (more reliable than WebSocket)
  const startSendingFrames = useCallback(() => {
    let isSending = false
    
    const sendFrame = async () => {
      if (!videoRef.current || !canvasRef.current) {
        animationFrameRef.current = requestAnimationFrame(sendFrame)
        return
      }

      // Skip if already sending (prevent overlapping requests)
      if (isSending) {
        animationFrameRef.current = requestAnimationFrame(sendFrame)
        return
      }

      try {
        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        // Check if video is ready
        if (video.readyState !== video.HAVE_ENOUGH_DATA) {
          animationFrameRef.current = requestAnimationFrame(sendFrame)
          return
        }

        // Performance optimization: Use lower resolution for faster processing
        const targetWidth = 640
        const targetHeight = 480
        canvas.width = targetWidth
        canvas.height = targetHeight
        ctx.drawImage(video, 0, 0, targetWidth, targetHeight)

        // Performance: Use lower quality for faster transmission
        const imageData = canvas.toDataURL('image/jpeg', 0.6)
        
        // Validate imageData before sending
        if (!imageData || typeof imageData !== 'string') {
          console.warn('⚠️ Invalid imageData: not a string')
          animationFrameRef.current = requestAnimationFrame(sendFrame)
          return
        }
        
        // Validate that it's a valid data URL with actual content
        if (!imageData.startsWith('data:image/')) {
          console.warn('⚠️ Invalid imageData: not a valid image data URL')
          animationFrameRef.current = requestAnimationFrame(sendFrame)
          return
        }
        
        // Check minimum length (data:image/jpeg;base64, is ~23 chars, need actual data)
        if (imageData.length < 100) {
          console.warn('⚠️ Invalid imageData: too short, likely empty')
          animationFrameRef.current = requestAnimationFrame(sendFrame)
          return
        }
        
        isSending = true
        setIsLoading(true)
        
        // Create AbortController for this request
        const controller = new AbortController()
        abortControllersRef.current.push(controller)
        
        try {
          console.log('📤 Sending frame to /analyze...')
          
          // Use improved API utility with timeout and retry
          const result = await analyzeFrame(imageData, (attempt, maxRetries, delay) => {
            console.log(`🔄 Retrying request (${attempt}/${maxRetries}) after ${delay}ms...`)
          }, controller)
          
          // Remove controller from tracking after successful request
          abortControllersRef.current = abortControllersRef.current.filter(c => c !== controller)
          
          console.log('✅ Received analysis data via HTTP:', {
            timestamp: result.timestamp || 'N/A',
            productivity: result.productivity?.productivity_score || 'N/A',
            hasPosture: !!result.posture,
            hasEyeStrain: !!result.eye_strain
          })
          
          // Use actual data from backend - no static defaults
          const safeResult = {
            timestamp: result.timestamp || new Date().toISOString(),
            posture: result.posture || { error: "No data available" },
            eye_strain: result.eye_strain || { error: "No data available" },
            engagement: result.engagement || { error: "No data available" },
            stress: result.stress || { error: "No data available" },
            productivity: result.productivity || { error: "Cannot calculate - missing data" },
            recommendations: result.recommendations || ["⚠️ Analysis in progress - collecting data..."]
          }
          setData(safeResult)
          setIsAnalyzing(true)
          setIsLoading(false)
          setError(null)
        } catch (fetchErr) {
          console.error('❌ Error sending frame via HTTP:', fetchErr)
          
          // Remove controller from tracking
          abortControllersRef.current = abortControllersRef.current.filter(c => c !== controller)
          
          setIsLoading(false)
          // Set user-friendly error message
          setError(fetchErr.message || 'Failed to analyze frame. Please try again.')
          // Don't clear data on error - keep showing last results
          setIsAnalyzing(false)
        } finally {
          isSending = false
          // Performance optimization: Throttle to ~5 FPS (200ms between frames) to reduce backend load
          setTimeout(() => {
            animationFrameRef.current = requestAnimationFrame(sendFrame)
          }, 200)
        }
      } catch (err) {
        console.error('❌ Error processing frame:', err)
        isSending = false
        setIsAnalyzing(false)
        stopSendingFrames()
      }
    }

    stopSendingFrames() // Ensure no duplicate loops
    animationFrameRef.current = requestAnimationFrame(sendFrame)
  }, [stopSendingFrames]) // Dependencies for useCallback

  // Function to connect WebSocket
  const connectWebSocket = useCallback(() => {
    // Clear any existing reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    try {
      console.log('🔌 Attempting to connect to WebSocket at ws://localhost:8000/ws...')
      
      // Close existing connection if any
      if (wsRef.current) {
        try {
          wsRef.current.close(1000, 'New connection attempt') // Clean close
        } catch (e) {
          console.log('Error closing existing connection:', e)
        }
      }
      
      const ws = new WebSocket('ws://localhost:8000/ws')
      wsRef.current = ws
      console.log('WebSocket object created, readyState:', ws.readyState)

      // Connection timeout - if not connected in 5 seconds, show error
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          console.error('❌ WebSocket connection timeout after 5 seconds')
          ws.close(1000, 'Connection timeout') // Clean close with reason
          setError('Connection timeout. Make sure backend is running on port 8000.')
          setIsAnalyzing(false)
        }
      }, 5000)

      ws.onopen = () => {
        clearTimeout(connectionTimeout)
        console.log('✅ WebSocket connected successfully!')
        console.log('WebSocket readyState:', ws.readyState)
        console.log('WebSocket URL:', ws.url)
        setIsAnalyzing(true)
        setError(null)
        console.log('Starting to send frames via HTTP POST...')
        // Use HTTP POST method instead of WebSocket for more reliability
        setTimeout(() => {
          startSendingFrames()
        }, 100)
      }

      ws.onmessage = (event) => {
        try {
          if (!event.data) {
            console.error('❌ Empty WebSocket message received')
            return
          }
          
          const response = JSON.parse(event.data)
          
          // Validate response
          if (!response || typeof response !== 'object') {
            console.error('❌ Invalid WebSocket response format:', response)
            setError('Invalid response from server')
            return
          }
          
          console.log('✅ Received analysis data:', {
            timestamp: response.timestamp || 'N/A',
            productivity: response.productivity?.productivity_score || 'N/A',
            hasRecommendations: response.recommendations?.length > 0,
            hasPosture: !!response.posture,
            hasEyeStrain: !!response.eye_strain,
            hasEngagement: !!response.engagement,
            hasStress: !!response.stress
          })
          
          // Use actual data from backend - no static defaults
          const safeResponse = {
            timestamp: response.timestamp || new Date().toISOString(),
            posture: response.posture || { error: "No data available" },
            eye_strain: response.eye_strain || { error: "No data available" },
            engagement: response.engagement || { error: "No data available" },
            stress: response.stress || { error: "No data available" },
            productivity: response.productivity || { error: "Cannot calculate - missing data" },
            recommendations: response.recommendations || ["⚠️ Analysis in progress - collecting data..."]
          }
          
          console.log('Full response:', safeResponse)
          setData(safeResponse)
          setIsAnalyzing(true) // Keep analyzing if we're receiving data
          setError(null) // Clear any previous errors
          console.log('✅ Data state updated, should display now')
        } catch (err) {
          console.error('❌ Error parsing WebSocket message:', err)
          console.error('Raw message:', event.data)
          setError(`Error processing server response: ${err.message}`)
        }
      }

      ws.onerror = (err) => {
        clearTimeout(connectionTimeout)
        console.error('❌ WebSocket error:', err)
        console.error('WebSocket state:', ws.readyState)
        console.error('WebSocket URL:', ws.url)
        setIsAnalyzing(false)
        setError('WebSocket connection error. Check: 1) Backend running on port 8000, 2) No firewall blocking, 3) Open http://localhost:8000/health in browser')
        stopSendingFrames()
      }

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout)
        console.log('🔌 WebSocket disconnected. Code:', event.code, 'Reason:', event.reason || 'No reason provided')
        console.log('Was clean close:', event.wasClean)
        console.log('Current data state:', data ? 'Has data' : 'No data')
        // Don't clear data on disconnect - keep showing last results
        setIsAnalyzing(false)
        stopSendingFrames()
        
        // Provide helpful error messages based on close code
        if (event.code === 1006) {
          console.error('❌ Connection closed abnormally (1006) - backend may not be running or connection refused')
          // Only set error if we don't have data to show
          if (!data) {
            setError('Connection failed. Backend may not be running. Check: http://localhost:8000/health')
          }
        } else if (event.code === 1002) {
          console.error('❌ Protocol error (1002)')
          if (!data) {
            setError('Protocol error. Check backend WebSocket implementation.')
          }
        } else if (event.code === 1003) {
          console.error('❌ Unsupported data (1003)')
          if (!data) {
            setError('Unsupported data format.')
          }
        } else if (event.code !== 1000 && event.code !== 1001) {
          console.error('❌ Unexpected close code:', event.code)
          if (!data) {
            setError(`Connection closed with code ${event.code}. Check backend logs.`)
          }
        }
        
        // Attempt to reconnect if the app is still "connected" (camera is on)
        // and the close was not a normal or explicit user-initiated close
        if (isConnectedRef.current && videoRef.current && event.code !== 1000 && event.code !== 1001) {
          console.log('🔄 Unexpected disconnect. Attempting to reconnect in 2 seconds...')
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isConnectedRef.current && videoRef.current && (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED)) {
              console.log('🔄 Reconnecting WebSocket...')
              if (connectWebSocketRef.current) {
                connectWebSocketRef.current()
              }
            }
          }, 2000)
        } else if (event.code === 1000) {
          console.log('WebSocket closed normally')
        }
      }
    } catch (err) {
      console.error('Error creating WebSocket:', err)
      setError('Failed to create WebSocket connection. Check console for details.')
      setIsAnalyzing(false)
      stopSendingFrames()
    }
  }, [startSendingFrames, stopSendingFrames]) // Dependencies for useCallback

  // Store connectWebSocket in ref so it can be called from other callbacks
  useEffect(() => {
    connectWebSocketRef.current = connectWebSocket
  }, [connectWebSocket])

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        }
      })
      
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        
        // Wait for video to be ready before connecting WebSocket (with timeout)
        const maxWaitTime = 5000 // 5 seconds max wait
        const checkInterval = 100 // Check every 100ms
        const startTime = Date.now()
        
        await new Promise((resolve, reject) => {
          const checkReady = () => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
              resolve()
            } else if (Date.now() - startTime > maxWaitTime) {
              reject(new Error('Video failed to become ready within timeout'))
            } else {
              setTimeout(checkReady, checkInterval)
            }
          }
          checkReady()
        })
      }
      
      setIsConnected(true)
      setSessionStartTime(Date.now())
      // Small delay to ensure video is fully ready and rendered before connecting WebSocket
      setTimeout(() => {
        if (connectWebSocketRef.current) {
          connectWebSocketRef.current()
        } else {
          connectWebSocket()
        }
      }, 500) // 500ms delay
    } catch (err) {
      let errorMessage = 'Camera access denied. Please allow camera permissions.'
      if (err.message && err.message.includes('timeout')) {
        errorMessage = 'Video failed to initialize. Please try again.'
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage = 'Camera access denied. Please allow camera permissions in your browser settings.'
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.'
      }
      setError(errorMessage)
      console.error('Camera error:', err)
      setIsConnected(false)
      setIsAnalyzing(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User stopped camera') // Clean close
    }
    stopSendingFrames()
    setIsConnected(false)
    setIsAnalyzing(false)
    setData(null)
    setError(null)
    setSessionStartTime(null)
    setSessionDuration(0)
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [stopSendingFrames])

  const getScoreColor = (score) => {
    if (score >= 80) return '#60a5fa' // Bright blue for excellent
    if (score >= 60) return '#3b82f6' // Medium blue for good
    return '#ef4444' // Red for poor
  }

  const getStatusIcon = (status) => {
    if (status === 'high' || status === 'low' && status !== 'low') return '✅'
    if (status === 'medium') return '⚠️'
    return '❌'
  }

  return (
    <div className="app">
      <header className="header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            <h1>🤖 AI Human-Computer Interaction Coach</h1>
            <p>Real-time wellness monitoring for your workspace</p>
          </div>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            {isConnected && sessionDuration > 0 && (
              <div style={{ 
                background: 'rgba(59, 130, 246, 0.2)', 
                padding: '8px 16px', 
                borderRadius: '8px',
                fontSize: '0.9rem',
                color: '#60a5fa'
              }}>
                ⏱️ {formatDuration(sessionDuration)}
              </div>
            )}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="theme-toggle"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="settings-button"
              title="Settings"
            >
              ⚙️
            </button>
          </div>
        </div>
      </header>

      {showSettings && (
        <div className="settings-panel">
          <div className="settings-content">
            <h3>Settings</h3>
            <div className="settings-item">
              <label>
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={(e) => setDarkMode(e.target.checked)}
                />
                Dark Mode
              </label>
            </div>
            <button onClick={() => setShowSettings(false)} className="close-settings">
              Close
            </button>
          </div>
        </div>
      )}

      <div className="container">
        <div className="camera-section">
          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video-preview"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            
            {!isConnected && (
              <div className="camera-overlay">
                <button onClick={startCamera} className="start-button">
                  🎥 Start Camera & Analysis
                </button>
              </div>
            )}

            {isConnected && (
              <div className="status-badge">
                {(() => {
                  if (!wsRef.current) return '🟡 Connecting...'
                  const state = wsRef.current.readyState
                  if (state === WebSocket.OPEN && isAnalyzing) return '🟢 Analyzing...'
                  if (state === WebSocket.OPEN) return '🟢 Connected'
                  if (state === WebSocket.CONNECTING) return '🟡 Connecting...'
                  if (state === WebSocket.CLOSING) return '🟠 Closing...'
                  return '🔴 Disconnected'
                })()}
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <p>Analyzing frame...</p>
          </div>
        )}

        {data && data.productivity && !data.productivity.error && (
          <div className="metrics-section">
            <div className="productivity-card">
              <h2>📊 Productivity Score</h2>
              <div 
                className="score-display"
                style={{ color: getScoreColor(data.productivity?.productivity_score || 0) }}
              >
                {data.productivity?.productivity_score ? data.productivity.productivity_score.toFixed(1) : 'N/A'}
              </div>
            </div>

            <div className="metrics-grid">
              {data.posture && !data.posture.error && (
                <div className="metric-card">
                  <h3>💺 Posture</h3>
                  {data.posture.score !== undefined && data.posture.score !== null ? (
                    <>
                      <div className="metric-value" style={{ color: getScoreColor(data.posture.score) }}>
                        {data.posture.score.toFixed(1)}
                      </div>
                      <div className="metric-status">
                        {data.posture.slouching ? '⚠️ Slouching detected' : '✅ Good posture'}
                      </div>
                    </>
                  ) : (
                    <div className="metric-status">⏳ Calibrating...</div>
                  )}
                </div>
              )}

              {data.eye_strain && !data.eye_strain.error && (
                <div className="metric-card">
                  <h3>👁️ Eye Strain</h3>
                  {data.eye_strain.score !== undefined && data.eye_strain.score !== null ? (
                    <>
                      <div className="metric-value" style={{ color: getScoreColor(data.eye_strain.score) }}>
                        {data.eye_strain.score.toFixed(1)}
                      </div>
                      <div className="metric-status">
                        Risk: {(data.eye_strain.eye_strain_risk || 'low').toUpperCase()}
                      </div>
                    </>
                  ) : (
                    <div className="metric-status">⏳ Calibrating...</div>
                  )}
                </div>
              )}

              {data.engagement && !data.engagement.error && (
                <div className="metric-card">
                  <h3>🧠 Engagement</h3>
                  {data.engagement.score !== undefined && data.engagement.score !== null ? (
                    <>
                      <div className="metric-value" style={{ color: getScoreColor(data.engagement.score) }}>
                        {data.engagement.score.toFixed(1)}
                      </div>
                      <div className="metric-status">
                        Concentration: {(data.engagement.concentration || 'low').toUpperCase()}
                      </div>
                    </>
                  ) : (
                    <div className="metric-status">⏳ Calibrating...</div>
                  )}
                </div>
              )}

              {data.stress && !data.stress.error && (
                <div className="metric-card">
                  <h3>😌 Stress Level</h3>
                  {data.stress.score !== undefined && data.stress.score !== null ? (
                    <>
                      <div className="metric-value" style={{ color: getScoreColor(data.stress.score) }}>
                        {data.stress.score.toFixed(1)}
                      </div>
                      <div className="metric-status">
                        {(data.stress.stress_level || 'low').toUpperCase()}
                      </div>
                    </>
                  ) : (
                    <div className="metric-status">⏳ Calibrating...</div>
                  )}
                </div>
              )}
            </div>

            {data.recommendations && data.recommendations.length > 0 && (
              <div className="recommendations-card">
                <h2>💡 Recommendations</h2>
                <ul className="recommendations-list">
                  {data.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              {isConnected && (
                <button onClick={stopCamera} className="stop-button">
                  🛑 Stop Analysis
                </button>
              )}
              {data && data.productivity && !data.productivity.error && (
                <button onClick={exportData} className="export-button">
                  📥 Export Data
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

