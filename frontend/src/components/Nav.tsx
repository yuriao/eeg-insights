import { Link, useLocation } from 'react-router-dom'

export default function Nav() {
  const loc = useLocation()
  const link = (to: string, label: string) => (
    <Link
      to={to}
      style={{
        fontFamily: 'sans-serif',
        fontSize: '0.85rem',
        fontWeight: 600,
        color: loc.pathname === to ? '#1a6b3c' : '#6b7280',
        borderBottom: loc.pathname === to ? '2px solid #1a6b3c' : '2px solid transparent',
        paddingBottom: 2,
      }}
    >
      {label}
    </Link>
  )

  return (
    <nav>
      <div>
        <Link to="/" className="nav-logo">🧠 EEG Insights</Link>
      </div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
        {link('/', 'Reports')}
        {link('/algorithms', 'Algorithms')}
        {link('/custom', 'Custom Analysis')}
      </div>
    </nav>
  )
}
