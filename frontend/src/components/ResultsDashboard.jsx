export default function ResultsDashboard({ results, originalImageUrl, onReset }) {
    const {
        predicted_bone_age_months,
        developmental_stage,
        gradcam_overlay_base64,
        processing_time_ms,
        chronological_age_months,
        delta_months
    } = results

    const years = Math.floor(predicted_bone_age_months / 12)
    const months = Math.round(predicted_bone_age_months % 12)

    const stageClass = developmental_stage?.toLowerCase() || 'child'

    // Delta classification
    const getDeltaClass = (delta) => {
        if (delta === undefined || delta === null) return ''
        const abs = Math.abs(delta)
        if (abs <= 12) return 'normal'
        if (abs <= 24) return 'mild'
        return 'significant'
    }

    const getDeltaValueClass = (delta) => {
        if (delta === undefined || delta === null) return ''
        if (Math.abs(delta) <= 12) return 'delta-value--normal'
        if (delta > 0) return 'delta-value--positive'
        return 'delta-value--negative'
    }

    const getDeltaLabel = (delta) => {
        if (delta === undefined || delta === null) return ''
        const abs = Math.abs(delta)
        if (abs <= 12) return 'Within normal range'
        if (abs <= 24) return 'Mild deviation'
        return 'Significant deviation — further evaluation recommended'
    }

    return (
        <div className="results">
            {/* Side-by-Side Image Viewer */}
            <div className="results__images">
                <div className="results__image-panel">
                    <span className="results__image-label">Original</span>
                    {originalImageUrl && (
                        <img
                            src={originalImageUrl}
                            alt="Original radiograph"
                            className="results__image"
                        />
                    )}
                </div>
                <div className="results__image-panel">
                    <span className="results__image-label">Grad-CAM Overlay</span>
                    {gradcam_overlay_base64 && (
                        <img
                            src={gradcam_overlay_base64}
                            alt="Grad-CAM heatmap overlay"
                            className="results__image"
                        />
                    )}
                </div>
            </div>

            {/* Metrics Cards */}
            <div className="results__metrics">
                {/* Predicted Bone Age */}
                <div className="metric-card">
                    <div className="metric-card__label">Predicted Bone Age</div>
                    <div className="metric-card__value">
                        {years}<span className="metric-card__unit">y</span>{' '}
                        {months}<span className="metric-card__unit">m</span>
                    </div>
                    <div className="metric-card__sub">
                        {predicted_bone_age_months} months
                    </div>
                </div>

                {/* Developmental Stage */}
                <div className="metric-card">
                    <div className="metric-card__label">Developmental Stage</div>
                    <div style={{ marginTop: '8px' }}>
                        <span className={`stage-badge stage-badge--${stageClass}`}>
                            {stageClass === 'child' && '👶'}
                            {stageClass === 'adolescent' && '🧑'}
                            {stageClass === 'adult' && '🧑‍🦳'}
                            {developmental_stage}
                        </span>
                    </div>
                </div>

                {/* Processing Time */}
                <div className="metric-card">
                    <div className="metric-card__label">Processing Time</div>
                    <div className="metric-card__value">
                        {processing_time_ms}<span className="metric-card__unit">ms</span>
                    </div>
                </div>

                {/* Delta (conditional) */}
                {delta_months !== undefined && delta_months !== null && (
                    <div className={`metric-card delta-card--${getDeltaClass(delta_months)}`}>
                        <div className="metric-card__label">Bone Age Delta</div>
                        <div className={`metric-card__value ${getDeltaValueClass(delta_months)}`}>
                            {delta_months > 0 ? '+' : ''}{delta_months}
                            <span className="metric-card__unit">months</span>
                        </div>
                        <div className="metric-card__sub">
                            {getDeltaLabel(delta_months)}
                        </div>
                    </div>
                )}
            </div>

            {/* Actions */}
            <div className="actions-bar">
                <button className="btn btn--secondary" onClick={onReset}>
                    🔄 New Analysis
                </button>
            </div>
        </div>
    )
}
