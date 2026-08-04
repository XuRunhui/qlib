import { useCallback, useEffect, useState } from 'react'
import Chart from './Chart'
import SearchBox from './SearchBox'
import Sectors from './Sectors'

const RANGES = ['1D', '1W', '1M', '3M', '1Y', 'MAX']
const LABEL = { '1D': 'Today', '1W': 'Past week', '1M': 'Past month', '3M': 'Past 3 months', '1Y': 'Past year', MAX: 'All time' }

export default function App() {
  const [view, setView] = useState('chart')
  const [ticker, setTicker] = useState('AAPL')
  const [range, setRange] = useState('1D')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hover, setHover] = useState(null)

  // The leaderboard has no custom-range endpoint, so fall back to a preset.
  const openSectors = () => {
    if (range === 'custom') setRange('1M')
    setView('sectors')
  }

  useEffect(() => {
    if (view === 'sectors') return
    const custom = range === 'custom'
    if (custom && !(from && to)) return
    const query = custom
      ? `ticker=${ticker}&from=${from}&to=${to}`
      : `ticker=${ticker}&range=${range}`

    let stale = false
    setLoading(true)
    setError(null)
    fetch(`/api/bars?${query}`)
      .then(async (r) => {
        const body = await r.json()
        if (!r.ok) throw new Error(body.detail || r.statusText)
        return body
      })
      .then((d) => { if (!stale) { setData(d); setHover(null) } })
      .catch((e) => { if (!stale) { setError(e.message); setData(null) } })
      .finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [view, ticker, range, from, to])

  const onHover = useCallback((point) => setHover(point), [])

  // The headline price is the official close on every range — the last point of
  // a series means something different in each aggregation (post-market print on
  // 1D, last continuous trade on 1W, closing auction on daily). Only the change
  // follows the selected range, measured from that range's baseline.
  // While the cursor is on the chart the header tracks it, like a broker app.
  const quote = data?.quote
  const price = hover ? hover.value : (quote?.close ?? data?.last)
  const change = data && price != null ? price - data.baseline : 0
  const changePct = data?.baseline ? (change / data.baseline) * 100 : 0
  const up = change >= 0
  const after = quote?.after

  const controls = (
    <div className="controls">
      <div className="pills">
        {RANGES.map((r) => (
          <button key={r} className={r === range ? 'pill on' : 'pill'} onClick={() => setRange(r)}>
            {r}
          </button>
        ))}
        {view === 'chart' && (
          <button className={range === 'custom' ? 'pill on' : 'pill'} onClick={() => setRange('custom')}>
            Custom
          </button>
        )}
      </div>

      {view === 'chart' && range === 'custom' && (
        <div className="dates">
          <input type="date" value={from} min="2021-08-02" onChange={(e) => setFrom(e.target.value)} />
          <span>→</span>
          <input type="date" value={to} min="2021-08-02" onChange={(e) => setTo(e.target.value)} />
        </div>
      )}
    </div>
  )

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Stock View</span>
        <nav className="views">
          <button className={view === 'chart' ? 'view on' : 'view'} onClick={() => setView('chart')}>Chart</button>
          <button className={view === 'sectors' ? 'view on' : 'view'} onClick={openSectors}>Sectors</button>
        </nav>
        <SearchBox onPick={(t) => { setTicker(t); setView('chart') }} />
      </header>

      {view === 'sectors' && (
        <>
          {controls}
          <Sectors range={range} onPick={(t) => { setTicker(t); setView('chart') }} />
        </>
      )}

      {view === 'chart' && error && <div className="error">{error}</div>}

      {view === 'chart' && data && (
        <>
          <div className="quote">
            <div className="sym">{data.ticker}</div>
            <div className="name">{data.name}</div>
            <div className="price">{price != null ? `$${price.toFixed(2)}` : '—'}</div>
            <div className={up ? 'delta up' : 'delta down'}>
              {up ? '+' : '−'}${Math.abs(change).toFixed(2)} ({up ? '+' : '−'}{Math.abs(changePct).toFixed(2)}%)
              <span className="period">
                {hover ? 'at cursor' : (LABEL[data.range] || `${from} → ${to}`)}
              </span>
            </div>

            {after && (
              <div className="after">
                ${after.price.toFixed(2)}
                <span className={after.change >= 0 ? 'up' : 'down'}>
                  {after.change >= 0 ? '+' : '−'}{Math.abs(after.change_pct).toFixed(2)}%
                </span>
                <span className="period">After hours</span>
              </div>
            )}
          </div>

          <div className={loading ? 'chart-wrap loading' : 'chart-wrap'}>
            <Chart data={data} onHover={onHover} />
          </div>
        </>
      )}

      {view === 'chart' && controls}

      <footer className="note">
        {data?.extended && 'Includes pre/post-market (4:00am–8:00pm ET). '}
        Prices are split-adjusted and delayed ~15 minutes. Dashed line ={' '}
        {data?.range === '1D' ? 'previous close' : 'first close in range'}.
      </footer>
    </div>
  )
}
