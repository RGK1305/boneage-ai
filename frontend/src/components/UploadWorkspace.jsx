import { useState, useRef, useCallback } from 'react'

export default function UploadWorkspace({ onAnalyze }) {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [sex, setSex] = useState('')
    const [chronologicalAge, setChronologicalAge] = useState('')
    const [isDragActive, setIsDragActive] = useState(false)
    const inputRef = useRef(null)

    const handleFile = useCallback((selectedFile) => {
        if (!selectedFile) return

        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
        if (!validTypes.includes(selectedFile.type)) {
            alert('Please upload a valid image file (JPEG, PNG, BMP, or TIFF)')
            return
        }

        const maxSize = 10 * 1024 * 1024 // 10MB
        if (selectedFile.size > maxSize) {
            alert('Image too large. Maximum size is 10MB.')
            return
        }

        setFile(selectedFile)
        const url = URL.createObjectURL(selectedFile)
        setPreview(url)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setIsDragActive(false)
        const droppedFile = e.dataTransfer.files[0]
        handleFile(droppedFile)
    }, [handleFile])

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragActive(true)
    }, [])

    const handleDragLeave = useCallback(() => {
        setIsDragActive(false)
    }, [])

    const handleFileInput = useCallback((e) => {
        handleFile(e.target.files[0])
    }, [handleFile])

    const handleSubmit = useCallback(() => {
        if (!file || !sex) return
        onAnalyze({
            file,
            sex,
            chronologicalAge: chronologicalAge ? parseFloat(chronologicalAge) : null
        })
    }, [file, sex, chronologicalAge, onAnalyze])

    const canSubmit = file && sex

    return (
        <div className="card" style={{ maxWidth: '640px', margin: '0 auto' }}>
            <div className="card__header">
                <h2 className="card__title">📤 Upload Radiograph</h2>
            </div>
            <div className="card__body">
                {/* Drop Zone */}
                <div
                    className={`upload-zone ${isDragActive ? 'upload-zone--active' : ''} ${file ? 'upload-zone--has-file' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/jpeg,image/png,image/bmp,image/tiff"
                        onChange={handleFileInput}
                        style={{ display: 'none' }}
                    />

                    {!file ? (
                        <>
                            <div className="upload-zone__icon">🩻</div>
                            <p className="upload-zone__text">
                                Drag & drop a hand radiograph or <strong>click to browse</strong>
                            </p>
                            <p className="upload-zone__hint">Supports JPEG, PNG, BMP, TIFF — Max 10MB</p>
                        </>
                    ) : (
                        <>
                            <div className="upload-zone__icon">✅</div>
                            <p className="upload-zone__filename">{file.name}</p>
                            {preview && <img src={preview} alt="Preview" className="upload-zone__preview" />}
                            <p className="upload-zone__hint" style={{ marginTop: '12px' }}>
                                Click to change image
                            </p>
                        </>
                    )}
                </div>

                {/* Sex Selection */}
                <div className="form-group" style={{ marginTop: '24px' }}>
                    <label className="form-label">
                        Biological Sex <span className="form-label__required">*</span>
                    </label>
                    <div className="radio-group">
                        <div className="radio-option">
                            <input
                                type="radio"
                                id="sex-male"
                                name="sex"
                                value="M"
                                checked={sex === 'M'}
                                onChange={(e) => setSex(e.target.value)}
                            />
                            <label htmlFor="sex-male" className="radio-option__label">♂ Male</label>
                        </div>
                        <div className="radio-option">
                            <input
                                type="radio"
                                id="sex-female"
                                name="sex"
                                value="F"
                                checked={sex === 'F'}
                                onChange={(e) => setSex(e.target.value)}
                            />
                            <label htmlFor="sex-female" className="radio-option__label">♀ Female</label>
                        </div>
                    </div>
                </div>

                {/* Optional Chronological Age */}
                <div className="form-group">
                    <label className="form-label" htmlFor="chrono-age">
                        Chronological Age (months) <span style={{ color: 'var(--color-text-tertiary)', fontWeight: 400 }}>— optional</span>
                    </label>
                    <input
                        id="chrono-age"
                        type="number"
                        className="form-input"
                        placeholder="e.g., 120"
                        min="0"
                        max="240"
                        value={chronologicalAge}
                        onChange={(e) => setChronologicalAge(e.target.value)}
                    />
                </div>

                {/* Submit */}
                <button
                    className="btn btn--primary btn--full"
                    disabled={!canSubmit}
                    onClick={handleSubmit}
                >
                    🔬 Analyze Radiograph
                </button>
            </div>
        </div>
    )
}
