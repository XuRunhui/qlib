import { useEffect, useRef, useState } from 'react'

export default function SearchBox({ onPick }) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState([])
  const [open, setOpen] = useState(false)
  const wrap = useRef(null)

  useEffect(() => {
    if (!q.trim()) return setHits([])
    const timer = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(q)}`)
        .then((r) => r.json())
        .then(setHits)
        .catch(() => setHits([]))
    }, 250)
    return () => clearTimeout(timer)
  }, [q])

  useEffect(() => {
    const close = (e) => !wrap.current?.contains(e.target) && setOpen(false)
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const pick = (symbol) => {
    onPick(symbol)
    setQ('')
    setOpen(false)
  }

  return (
    <div className="search" ref={wrap}>
      <input
        value={q}
        placeholder="Search symbol or company"
        onChange={(e) => { setQ(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => e.key === 'Enter' && q.trim() && pick(q.trim().toUpperCase())}
      />
      {open && hits.length > 0 && (
        <ul className="hits">
          {hits.map((h) => (
            <li key={h.ticker} onClick={() => pick(h.ticker)}>
              <span className="hit-sym">{h.ticker}</span>
              <span className="hit-name">{h.name}</span>
              <span className="hit-exch">{h.exchange}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
