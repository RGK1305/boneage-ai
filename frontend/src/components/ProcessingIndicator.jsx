export default function ProcessingIndicator() {
    return (
        <div className="card" style={{ maxWidth: '640px', margin: '0 auto' }}>
            <div className="processing">
                <div className="processing__spinner" />
                <p className="processing__text">Analyzing radiograph features…</p>
                <p className="processing__subtext">
                    Running deep feature extraction and ensemble prediction
                </p>
            </div>
        </div>
    )
}
