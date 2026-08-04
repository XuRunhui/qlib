import { useEffect, useRef } from 'react'
import { AreaSeries, CrosshairMode, LineStyle, createChart } from 'lightweight-charts'

const UP = '#00c805'
const DOWN = '#ff5000'

// Bar timestamps are true UTC seconds; lightweight-charts labels them UTC by
// default, so every formatter below re-renders them in market time.
const etTime = (t) =>
  new Date(t * 1000).toLocaleTimeString('en-US', {
    timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit',
  })

const etDate = (t, withYear) =>
  new Date(t * 1000).toLocaleDateString('en-US', {
    timeZone: 'America/New_York', month: 'short', day: 'numeric',
    ...(withYear ? { year: 'numeric' } : {}),
  })

export default function Chart({ data, onHover }) {
  const box = useRef(null)

  useEffect(() => {
    if (!data) return
    const intraday = data.timespan === 'minute'
    const singleDay = data.range === '1D'
    // Same reference price the header uses, so curve colour and header agree.
    const ref = data.quote?.close ?? data.last
    const color = ref - data.baseline >= 0 ? UP : DOWN

    const chart = createChart(box.current, {
      width: box.current.clientWidth,
      height: box.current.clientHeight,
      layout: { background: { color: 'transparent' }, textColor: '#9b9b9b', attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.15, bottom: 0.15 } },
      timeScale: {
        borderVisible: false,
        timeVisible: intraday,
        secondsVisible: false,
        tickMarkFormatter: (t) => (singleDay ? etTime(t) : etDate(t, data.range === 'MAX')),
      },
      localization: {
        priceFormatter: (p) => `$${p.toFixed(2)}`,
        timeFormatter: (t) => (intraday ? `${etDate(t)} ${etTime(t)} ET` : etDate(t, true)),
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: { color: '#666', width: 1, style: LineStyle.Solid, labelBackgroundColor: '#333' },
        horzLine: { color: '#666', width: 1, style: LineStyle.Solid, labelBackgroundColor: '#333' },
      },
      handleScale: false,
      handleScroll: false,
    })

    const series = chart.addSeries(AreaSeries, {
      lineColor: color,
      lineWidth: 2,
      topColor: `${color}33`,
      bottomColor: `${color}00`,
      priceLineVisible: false,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: color,
      crosshairMarkerBackgroundColor: color,
    })
    series.setData(data.bars.map((b) => ({ time: b.t / 1000, value: b.c })))

    // Dashed reference at the prior close (1D) or the period's first close.
    series.createPriceLine({
      price: data.baseline,
      color: '#666',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: '',
    })

    chart.timeScale().fitContent()

    chart.subscribeCrosshairMove((param) => {
      const point = param.seriesData.get(series)
      onHover(point ? { value: point.value, time: param.time } : null)
    })

    const resize = new ResizeObserver(() =>
      chart.applyOptions({ width: box.current.clientWidth, height: box.current.clientHeight }),
    )
    resize.observe(box.current)

    return () => {
      resize.disconnect()
      chart.remove()
    }
  }, [data, onHover])

  return <div className="chart" ref={box} />
}
