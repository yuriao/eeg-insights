import { Link } from 'react-router-dom'

export default function Nav() {
  return (
    <nav>
      <div>
        <Link to="/" className="nav-logo">⚡ EEG Insights</Link>
      </div>
      <div className="nav-tagline">Automated EEG analysis reports</div>
    </nav>
  )
}
