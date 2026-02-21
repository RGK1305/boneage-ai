import { useState, useCallback } from 'react'
import UploadWorkspace from './components/UploadWorkspace'
import ProcessingIndicator from './components/ProcessingIndicator'
import ResultsDashboard from './components/ResultsDashboard'

const API_BASE = 'http://localhost:8000'

function App() {
  // States: idle | processing | results | error
  const [appState, setAppState] = useState('idle')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [originalImageUrl, setOriginalImageUrl] = useState(null)

  const handleAnalyze = useCallback(async ({ file, sex, chronologicalAge }) => {
    setAppState('processing')
    setError(null)

    // Create preview URL for original image
    const imageUrl = URL.createObjectURL(file)
    setOriginalImageUrl(imageUrl)

    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('patient_sex', sex)
      if (chronologicalAge !== null && chronologicalAge !== '') {
        formData.append('chronological_age_months', chronologicalAge)
      }

      const response = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
      setAppState('results')
    } catch (err) {
      setError(err.message || 'An unexpected error occurred')
      setAppState('error')
    }
  }, [])

  const handleReset = useCallback(() => {
    setAppState('idle')
    setResults(null)
    setError(null)
    if (originalImageUrl) {
      URL.revokeObjectURL(originalImageUrl)
      setOriginalImageUrl(null)
    }
  }, [originalImageUrl])

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-header__logo">
          <div className="app-header__icon">🦴</div>
          <div>
            <div className="app-header__title">BoneAge AI</div>
            <div className="app-header__subtitle">Automated Bone Age Assessment</div>
          </div>
        </div>
      </header>

      <main className="app-main">
        {error && (
          <div className="error-message">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {(appState === 'idle' || appState === 'error') && (
          <UploadWorkspace onAnalyze={handleAnalyze} />
        )}

        {appState === 'processing' && (
          <ProcessingIndicator />
        )}

        {appState === 'results' && results && (
          <ResultsDashboard
            results={results}
            originalImageUrl={originalImageUrl}
            onReset={handleReset}
          />
        )}
      </main>
    </div>
  )
}

export default App
