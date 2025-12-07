import React, { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const animationFrameRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const isConnectedRef = useRef(false) // Use ref to avoid stale closures
  const connectWebSocketRef = useRef(null) // Ref to store connectWebSocket function

  useEffect(() => {
    isConnectedRef.current = isConnected
  }, [isConnected])

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

        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        ctx.drawImage(video, 0, 0)

        const imageData = canvas.toDataURL('image/jpeg', 0.7)
        
        isSending = true
        try {
          console.log('📤 Sending frame to /analyze...')
          // Use HTTP POST instead of WebSocket
          // Use relative URL to go through Vite proxy (avoids CORS issues)
          const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
          })
          console.log('📥 Response received:', response.status, response.statusText)
          
          if (response.ok) {
            const result = await response.json()
            console.log('✅ Received analysis data via HTTP:', {
              timestamp: result.timestamp,
              productivity: result.productivity?.productivity_score
            })
            setData(result)
            setIsAnalyzing(true)
            setError(null)
          } else {
            const errorText = await response.text()
            console.error('❌ HTTP request failed:', {
              status: response.status,
              statusText: response.statusText,
              body: errorText
            })
            setError(`Analysis failed: ${response.status} ${response.statusText}`)
          }
        } catch (fetchErr) {
          console.error('❌ Error sending frame via HTTP:', fetchErr)
          console.error('Error details:', {
            message: fetchErr.message,
            name: fetchErr.name,
            type: fetchErr.constructor.name
          })
          
          // Check if it's a timeout
          if (fetchErr.name === 'TimeoutError' || fetchErr.name === 'AbortError') {
            setError('Request timeout. Backend may be slow or unresponsive.')
          } else if (fetchErr.message.includes('Failed to fetch') || fetchErr.message.includes('NetworkError') || fetchErr.message.includes('ERR_CONNECTION_REFUSED')) {
            setError('Cannot connect to backend. Make sure backend is running on http://localhost:8000')
          } else {
            setError(`Connection error: ${fetchErr.message}`)
          }
        } finally {
          isSending = false
          // Throttle to ~5 FPS (200ms between frames) to reduce backend load
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
          const response = JSON.parse(event.data)
          console.log('✅ Received analysis data:', {
            timestamp: response.timestamp,
            productivity: response.productivity?.productivity_score,
            hasRecommendations: response.recommendations?.length > 0,
            hasPosture: !!response.posture,
            hasEyeStrain: !!response.eye_strain,
            hasEngagement: !!response.engagement,
            hasStress: !!response.stress
          })
          console.log('Full response:', response)
          setData(response)
          setIsAnalyzing(true) // Keep analyzing if we're receiving data
          setError(null) // Clear any previous errors
          console.log('✅ Data state updated, should display now')
        } catch (err) {
          console.error('❌ Error parsing WebSocket message:', err)
          console.error('Raw message:', event.data)
          setError('Error processing server response')
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

  const startCamera = async () => {
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
        
        // Wait for video to be ready before connecting WebSocket
        await new Promise((resolve) => {
          const checkReady = () => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
              resolve()
            } else {
              setTimeout(checkReady, 100)
            }
          }
          checkReady()
        })
      }
      
      setIsConnected(true)
      // Small delay to ensure video is fully ready and rendered before connecting WebSocket
      setTimeout(() => {
        if (connectWebSocketRef.current) {
          connectWebSocketRef.current()
        } else {
          connectWebSocket()
        }
      }, 500) // 500ms delay
    } catch (err) {
      setError('Camera access denied. Please allow camera permissions.')
      console.error('Camera error:', err)
      setIsConnected(false)
      setIsAnalyzing(false)
    }
  }

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
        <h1>🤖 AI Human-Computer Interaction Coach</h1>
        <p>Real-time wellness monitoring for your workspace</p>
      </header>

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

        {data && data.productivity && (
          <div className="metrics-section">
            <div className="productivity-card">
              <h2>📊 Productivity Score</h2>
              <div 
                className="score-display"
                style={{ color: getScoreColor(data.productivity?.productivity_score || 0) }}
              >
                {(data.productivity?.productivity_score || 0).toFixed(1)}
              </div>
            </div>

            <div className="metrics-grid">
              {data.posture && (
                <div className="metric-card">
                  <h3>💺 Posture</h3>
                  <div className="metric-value" style={{ color: getScoreColor(data.posture?.score || 0) }}>
                    {(data.posture?.score || 0).toFixed(1)}
                  </div>
                  <div className="metric-status">
                    {data.posture?.slouching ? '⚠️ Slouching detected' : '✅ Good posture'}
                  </div>
                </div>
              )}

              {data.eye_strain && (
                <div className="metric-card">
                  <h3>👁️ Eye Strain</h3>
                  <div className="metric-value" style={{ color: getScoreColor(data.eye_strain?.score || 0) }}>
                    {(data.eye_strain?.score || 0).toFixed(1)}
                  </div>
                  <div className="metric-status">
                    Risk: {(data.eye_strain?.eye_strain_risk || 'low').toUpperCase()}
                  </div>
                </div>
              )}

              {data.engagement && (
                <div className="metric-card">
                  <h3>🧠 Engagement</h3>
                  <div className="metric-value" style={{ color: getScoreColor(data.engagement?.score || 0) }}>
                    {(data.engagement?.score || 0).toFixed(1)}
                  </div>
                  <div className="metric-status">
                    Concentration: {(data.engagement?.concentration || 'low').toUpperCase()}
                  </div>
                </div>
              )}

              {data.stress && (
                <div className="metric-card">
                  <h3>😌 Stress Level</h3>
                  <div className="metric-value" style={{ color: getScoreColor(data.stress?.score || 0) }}>
                    {(data.stress?.score || 0).toFixed(1)}
                  </div>
                  <div className="metric-status">
                    {(data.stress?.stress_level || 'low').toUpperCase()}
                  </div>
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

            {isConnected && (
              <button onClick={stopCamera} className="stop-button">
                🛑 Stop Analysis
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App

