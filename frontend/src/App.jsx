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

  // Function to send frames
  const startSendingFrames = useCallback(() => {
    const sendFrame = () => {
      if (!videoRef.current || !canvasRef.current) {
        animationFrameRef.current = requestAnimationFrame(sendFrame)
        return
      }

      // Check WebSocket state
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        // If WebSocket is not open, stop sending frames
        stopSendingFrames()
        setIsAnalyzing(false)
        // Reconnection logic is handled in ws.onclose
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

        const imageData = canvas.toDataURL('image/jpeg', 0.8)
        
        wsRef.current.send(JSON.stringify({
          image: imageData
        }))

        animationFrameRef.current = requestAnimationFrame(sendFrame)
      } catch (err) {
        console.error('❌ Error sending frame:', err)
        setError('Failed to send video frame. Connection may be broken.')
        setIsAnalyzing(false)
        stopSendingFrames()
        // Don't reconnect here - let onclose handle it
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
        console.log('Starting to send frames...')
        // Small delay before starting to send frames to ensure connection is stable
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            startSendingFrames()
          } else {
            console.warn('WebSocket not open when trying to start sending frames')
          }
        }, 100)
      }

      ws.onmessage = (event) => {
        try {
          const response = JSON.parse(event.data)
          setData(response)
          setIsAnalyzing(true) // Keep analyzing if we're receiving data
          setError(null) // Clear any previous errors
          console.log('✅ Received analysis data')
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
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
        setIsAnalyzing(false)
        stopSendingFrames()
        
        // Provide helpful error messages based on close code
        if (event.code === 1006) {
          console.error('❌ Connection closed abnormally (1006) - backend may not be running or connection refused')
          setError('Connection failed. Backend may not be running. Check: http://localhost:8000/health')
        } else if (event.code === 1002) {
          console.error('❌ Protocol error (1002)')
          setError('Protocol error. Check backend WebSocket implementation.')
        } else if (event.code === 1003) {
          console.error('❌ Unsupported data (1003)')
          setError('Unsupported data format.')
        } else if (event.code !== 1000 && event.code !== 1001) {
          console.error('❌ Unexpected close code:', event.code)
          setError(`Connection closed with code ${event.code}. Check backend logs.`)
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
    if (score >= 80) return '#10b981'
    if (score >= 60) return '#f59e0b'
    return '#ef4444'
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

        {data && (
          <div className="metrics-section">
            <div className="productivity-card">
              <h2>📊 Productivity Score</h2>
              <div 
                className="score-display"
                style={{ color: getScoreColor(data.productivity.productivity_score) }}
              >
                {data.productivity.productivity_score.toFixed(1)}
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <h3>💺 Posture</h3>
                <div className="metric-value" style={{ color: getScoreColor(data.posture.score) }}>
                  {data.posture.score.toFixed(1)}
                </div>
                <div className="metric-status">
                  {data.posture.slouching ? '⚠️ Slouching detected' : '✅ Good posture'}
                </div>
              </div>

              <div className="metric-card">
                <h3>👁️ Eye Strain</h3>
                <div className="metric-value" style={{ color: getScoreColor(data.eye_strain.score) }}>
                  {data.eye_strain.score.toFixed(1)}
                </div>
                <div className="metric-status">
                  Risk: {data.eye_strain.eye_strain_risk.toUpperCase()}
                </div>
              </div>

              <div className="metric-card">
                <h3>🧠 Engagement</h3>
                <div className="metric-value" style={{ color: getScoreColor(data.engagement.score) }}>
                  {data.engagement.score.toFixed(1)}
                </div>
                <div className="metric-status">
                  Concentration: {data.engagement.concentration.toUpperCase()}
                </div>
              </div>

              <div className="metric-card">
                <h3>😌 Stress Level</h3>
                <div className="metric-value" style={{ color: getScoreColor(data.stress.score) }}>
                  {data.stress.score.toFixed(1)}
                </div>
                <div className="metric-status">
                  {data.stress.stress_level.toUpperCase()}
                </div>
              </div>
            </div>

            <div className="recommendations-card">
              <h2>💡 Recommendations</h2>
              <ul className="recommendations-list">
                {data.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>

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

