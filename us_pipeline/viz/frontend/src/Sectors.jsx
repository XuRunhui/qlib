import { useEffect, useMemo, useState } from 'react'

const COLUMNS = [
  { key: 'ticker', label: 'Symbol', align: 'left' },
  { key: 'industry', label: 'Industry', align: 'left' },
  { key: 'price', label: 'Price', align: 'right' },
  { key: 'change_pct', label: 'Change', align: 'right' },
  { key: 'volume', label: 'Volume', align: 'right' },
]

const UNIVERSES = [['sp500', 'S&P 500'], ['all', 'All stocks']]

const fmtVol = (v) =>
  v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`

export default function Sectors({ range, onPick }) {
  const [data, setData] = useState(null)
  const [universe, setUniverse] = useState('sp500')
  const [liquid, setLiquid] = useState(true)
  const [sector, setSector] = useState('All')
  const [sort, setSort] = useState({ key: 'change_pct', dir: -1 })
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let stale = false
    setLoading(true)
    setError(null)
    fetch(`/api/movers?range=${range}&universe=${universe}&liquid=${liquid}`)
      .then(async (r) => {
        const body = await r.json()
        if (!r.ok) throw new Error(body.detail || r.statusText)
        return body
      })
      .then((d) => { if (!stale) setData(d) })
      .catch((e) => { if (!stale) { setError(e.message); setData(null) } })
      .finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [range, universe, liquid])

  const tabs = useMemo(() => {
    if (!data) return []
    const counts = {}
    for (const r of data.rows) counts[r.sector] = (counts[r.sector] || 0) + 1
    return [['All', data.rows.length], ...Object.entries(counts).sort((a, b) => b[1] - a[1])]
  }, [data])

  // Switching universe can remove the selected sector (ETFs only exist in the
  // wide one), so fall back to All rather than showing an empty table.
  useEffect(() => {
    if (data && sector !== 'All' && !data.rows.some((r) => r.sector === sector)) setSector('All')
  }, [data, sector])

  const rows = useMemo(() => {
    if (!data) return []
    const kept = sector === 'All' ? data.rows : data.rows.filter((r) => r.sector === sector)
    return [...kept].sort((a, b) => (a[sort.key] > b[sort.key] ? 1 : a[sort.key] < b[sort.key] ? -1 : 0) * sort.dir)
  }, [data, sector, sort])

  const clickHeader = (key) =>
    setSort((s) => (s.key === key ? { key, dir: -s.dir } : { key, dir: key === 'ticker' || key === 'industry' ? 1 : -1 }))

  const picker = (
    <div className="universe">
      <div className="seg">
        {UNIVERSES.map(([key, label]) => (
          <button key={key} className={universe === key ? 'seg-btn on' : 'seg-btn'} onClick={() => setUniverse(key)}>
            {label}
          </button>
        ))}
      </div>
      <label className="check">
        <input type="checkbox" checked={liquid} onChange={(e) => setLiquid(e.target.checked)} />
        Liquid only
        <span className="hint">price &gt; $5, volume &gt; $10M</span>
      </label>
    </div>
  )

  if (error) return <div className="sectors">{picker}<div className="error">{error}</div></div>
  if (!data) return <div className="sectors">{picker}<div className="note">Loading…</div></div>

  return (
    <div className={loading ? 'sectors loading' : 'sectors'}>
      {picker}
      <div className="tabs">
        {tabs.map(([name, count]) => (
          <button key={name} className={name === sector ? 'tab on' : 'tab'} onClick={() => setSector(name)}>
            {name} <span className="tab-count">{count}</span>
          </button>
        ))}
      </div>

      <div className="window">
        {data.rows.length} names · {data.from} → {data.to}
        {data.clamped && ' · window shortened to the earliest date the grouped feed serves'}
        {data.illiquid > 0 && ` · ${data.illiquid} below the liquidity floor`}
        {data.renamed > 0 && ` · ${data.renamed} changed ticker in this window`}
      </div>

      <table className="movers">
        <thead>
          <tr>
            {COLUMNS.map((c) => (
              <th
                key={c.key}
                className={c.align === 'right' ? 'right' : ''}
                onClick={() => clickHeader(c.key)}
              >
                {c.label}
                {sort.key === c.key && <span className="arrow">{sort.dir < 0 ? '▼' : '▲'}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.ticker} onClick={() => onPick(r.ticker)}>
              <td className="sym-cell">{r.ticker}</td>
              {/* ETFs have no SIC industry, so fall back to the fund name. */}
              <td className="industry-cell">{r.industry || r.name}</td>
              <td className="right">${r.price.toFixed(2)}</td>
              <td className={r.change_pct >= 0 ? 'right up' : 'right down'}>
                {r.change_pct >= 0 ? '+' : '−'}{Math.abs(r.change_pct).toFixed(2)}%
              </td>
              <td className="right muted">{fmtVol(r.volume)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
