#!/usr/bin/env python3
"""
backtest.py - Backtest de estrategia HOLD con TP/SL dinámicos para Binance o KuCoin.
Incluye análisis de capital con interés compuesto (reinversión total).

Uso: python backtest.py --exchange binance --symbol SOLUSDT --timeframe 1 --days 7 --initial_capital 1000
      python backtest.py --exchange kucoin --timeframe 1 --days 7   # top 20 assets
"""

import argparse
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
ATR_PERIOD = 14
ADX_PERIOD = 14
TP_MULT = 1.0
SL_MULT = 1.0
MAX_ASSETS = 20

# ==============================
# FUNCIONES COMUNES (INDICADORES)
# ==============================
def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    up = df['high'].diff()
    down = df['low'].diff() * -1
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / tr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    return pd.Series(dx).rolling(period).mean()

def slope(df, period=3):
    return df['close'].diff(period)

# ==============================
# FUNCIONES DE DESCARGA POR EXCHANGE
# ==============================
def get_top_assets_binance():
    """Top 20 por volumen en USDT (Binance)"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df = df.sort_values('quoteVolume', ascending=False)
    top = df['symbol'].tolist()
    top = [s for s in top if s.endswith('USDT')]
    return top[:MAX_ASSETS]

def fetch_klines_binance(symbol, interval, days=7):
    """Descarga velas de Binance (intervalo en minutos: '1m', '3m', '5m')"""
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_klines = []
    limit = 1000
    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"Error {resp.status_code} - {symbol}")
            break
        klines = resp.json()
        if not klines:
            break
        all_klines.extend(klines)
        start_time = klines[-1][6] + 1
        time.sleep(0.1)
    cols = ['open_time','open','high','low','close','volume','close_time','quote_vol','trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    df = df[['open_time','open','high','low','close','volume']]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df.rename(columns={'open_time':'timestamp'}, inplace=True)
    return df

def get_top_assets_kucoin():
    """Top 20 por volumen en USDT (KuCoin), excluyendo stablecoins"""
    url = "https://api.kucoin.com/api/v1/market/allTickers"
    data = requests.get(url).json()
    if data['code'] != '200000':
        raise Exception("Error obteniendo tickers de KuCoin")
    tickers = data['data']['ticker']
    df = pd.DataFrame(tickers)
    df['volValue'] = df['volValue'].astype(float)
    df = df.sort_values('volValue', ascending=False)
    stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'PAX', 'HUSD', 'USDN', 'FEI', 'FRAX', 'LUSD', 'MIM', 'ALUSD', 'DOLA', 'OUSD', 'USDX', 'EURS', 'EURT', 'EURL', 'CEUR', 'CUSD', 'XUSD', 'UST', 'USTC']
    top = []
    for _, row in df.iterrows():
        symbol = row['symbol']
        if symbol.endswith('-USDT'):
            base = symbol.replace('-USDT', '')
            if base not in stablecoins:
                top.append(symbol)
        if len(top) >= MAX_ASSETS:
            break
    return top

def fetch_klines_kucoin(symbol, interval, days=7):
    """
    Descarga velas de KuCoin.
    interval: 1, 3, 5 (minutos)
    """
    interval_map = {1: "1min", 3: "3min", 5: "5min"}
    kucoin_interval = interval_map.get(interval, "1min")
    base_url = "https://api.kucoin.com/api/v1/market/candles"
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=days)).timestamp())
    all_klines = []
    limit = 1500
    while start < end:
        params = {
            'symbol': symbol,
            'type': kucoin_interval,
            'startAt': start,
            'endAt': end
        }
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"Error {resp.status_code} - {symbol}")
            break
        data = resp.json()
        if data['code'] != '200000' or not data['data']:
            break
        klines = data['data']
        all_klines.extend(klines)
        last_time = int(klines[-1][0])
        start = last_time + 1
        time.sleep(0.2)
    # KuCoin devuelve [time, open, close, high, low, volume, turnover]
    df = pd.DataFrame(all_klines, columns=['timestamp','open','close','high','low','volume','turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp','open','high','low','close','volume']]

# ==============================
# ESTRATEGIA CON ANÁLISIS DE CAPITAL
# ==============================
class StrategyHold:
    def __init__(self, df, initial_capital=1000):
        self.df = df.copy()
        self.initial_capital = initial_capital
        # Calcular indicadores
        self.df['ATR'] = atr(df, ATR_PERIOD)
        self.df['ADX'] = adx(df, ADX_PERIOD)
        self.df['Slope'] = slope(df, 3)

    def signal(self, i):
        """Genera señal en la vela i (comparando cierre actual con anterior)"""
        if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
            return "LONG"
        elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
            return "SHORT"
        return None

    def dynamic_levels(self, i, entry, signal):
        """Calcula TP y SL basados en ATR y ADX de la vela i (momento de entrada)"""
        atr_val = self.df['ATR'].iloc[i]
        adx_val = self.df['ADX'].iloc[i]
        sl_mult = SL_MULT * (2.0 if adx_val > 25 else 1.0)
        tp_mult = TP_MULT * (1.0 if adx_val > 25 else 0.5)
        if signal == "LONG":
            sl = entry - atr_val * sl_mult
            tp = entry + atr_val * tp_mult
        else:
            sl = entry + atr_val * sl_mult
            tp = entry - atr_val * tp_mult
        return tp, sl

    def backtest(self):
        """
        Ejecuta el backtest.
        Retorna:
            trades: lista de resultados en precio (diferencia entry-exit)
            capital_series: lista de capital después de cada trade (incluye el inicial antes del primer trade)
            returns: lista de retornos porcentuales de cada trade
        """
        trades = []          # resultado en precio
        returns = []         # retorno porcentual de cada trade
        capital = self.initial_capital
        capital_series = [capital]  # capital antes del primer trade

        open_positions = []  # cada elemento: {'entry': precio, 'tp': tp, 'sl': sl, 'signal': signal}

        for i in range(1, len(self.df)):
            # 1. Generar señal al cierre de la vela i
            sig = self.signal(i)
            if sig:
                entry = self.df['close'].iloc[i]
                tp, sl = self.dynamic_levels(i, entry, sig)
                open_positions.append({'entry': entry, 'tp': tp, 'sl': sl, 'signal': sig})

            # 2. Verificar cierres de posiciones abiertas con la vela i (high/low)
            remaining = []
            for pos in open_positions:
                high = self.df['high'].iloc[i]
                low = self.df['low'].iloc[i]
                trade_result = None
                if pos['signal'] == "LONG":
                    if high >= pos['tp']:
                        trade_result = pos['tp'] - pos['entry']
                    elif low <= pos['sl']:
                        trade_result = pos['sl'] - pos['entry']
                else:  # SHORT
                    if low <= pos['tp']:
                        trade_result = pos['entry'] - pos['tp']
                    elif high >= pos['sl']:
                        trade_result = pos['entry'] - pos['sl']

                if trade_result is not None:
                    trades.append(trade_result)
                    # Calcular retorno porcentual y actualizar capital
                    ret = trade_result / pos['entry']
                    returns.append(ret)
                    capital *= (1 + ret)
                    capital_series.append(capital)
                else:
                    remaining.append(pos)
            open_positions = remaining

        return trades, returns, capital_series

# ==============================
# MÉTRICAS Y REPORTE
# ==============================
def metrics(trades, returns, capital_series):
    trades = np.array(trades)
    returns = np.array(returns)
    if len(trades) == 0:
        return {
            "Winrate %": 0,
            "Profit Factor": 0,
            "MAE": 0,
            "MFE": 0,
            "MSI": 0,
            "Total Trades": 0,
            "Initial Capital": capital_series[0],
            "Final Capital": capital_series[0],
            "Total Return %": 0,
            "CAGR %": 0
        }
    winrate = (trades > 0).sum() / len(trades) * 100
    pf = trades[trades > 0].sum() / abs(trades[trades < 0].sum()) if (trades < 0).any() else np.inf
    mae = abs(trades[trades < 0]).mean() if (trades < 0).any() else 0
    mfe = trades[trades > 0].mean() if (trades > 0).any() else 0
    msi = mfe / (mae + 1e-8)

    initial_cap = capital_series[0]
    final_cap = capital_series[-1]
    total_return_pct = (final_cap - initial_cap) / initial_cap * 100

    # Estimar días de trading (aproximado a partir de la longitud de la serie de capital)
    # Podríamos usar la longitud de velas, pero no tenemos acceso aquí. Por simplicidad, omitimos CAGR.
    # Si se desea, se puede pasar el número de días como parámetro.
    cagr = 0  # No calculado

    return {
        "Winrate %": winrate,
        "Profit Factor": pf,
        "MAE": mae,
        "MFE": mfe,
        "MSI": msi,
        "Total Trades": len(trades),
        "Initial Capital": initial_cap,
        "Final Capital": final_cap,
        "Total Return %": total_return_pct,
        "CAGR %": cagr
    }

def generate_report(symbol, tf, trades, returns, capital_series):
    stats = metrics(trades, returns, capital_series)
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"{symbol} - Timeframe {tf} min")
    lines.append(f"{'='*60}")
    lines.append(f"Trades cerrados        : {stats['Total Trades']}")
    lines.append(f"Winrate %              : {stats['Winrate %']:.2f}")
    lines.append(f"Profit Factor          : {stats['Profit Factor']:.2f}")
    lines.append(f"MAE (pérdida media)    : {stats['MAE']:.6f}")
    lines.append(f"MFE (ganancia media)   : {stats['MFE']:.6f}")
    lines.append(f"MSI                    : {stats['MSI']:.2f}")
    lines.append(f"Capital inicial        : ${stats['Initial Capital']:.2f}")
    lines.append(f"Capital final          : ${stats['Final Capital']:.2f}")
    lines.append(f"Retorno total          : {stats['Total Return %']:.2f}%")
    if stats['Total Trades'] > 0:
        lines.append(f"\nEvolución del capital (cada 10 trades):")
        for idx, cap in enumerate(capital_series):
            if idx % 10 == 0 or idx == len(capital_series)-1:
                lines.append(f"  Trade {idx}: ${cap:.2f}")
        # También mostrar primeros 200 trades como símbolos (+/-)
        show = trades[:200]
        bars = ''.join(['+' if t>0 else '-' if t<0 else '.' for t in show])
        lines.append(f"\nPrimeros {len(show)} trades:\n{bars}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)

# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Backtest HOLD con TP/SL dinámicos y análisis de capital')
    parser.add_argument('--exchange', choices=['binance', 'kucoin'], required=True, help='Exchange a utilizar')
    parser.add_argument('--symbol', help='Símbolo (ej. SOLUSDT para Binance, SOL-USDT para KuCoin). Si no se indica, se toman los top 20 del exchange.')
    parser.add_argument('--timeframe', type=int, default=1, choices=[1,3,5], help='Temporalidad en minutos')
    parser.add_argument('--days', type=int, default=7, help='Días de historia a descargar')
    parser.add_argument('--initial_capital', type=float, default=1000.0, help='Capital inicial para la simulación')
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    # Determinar lista de símbolos a procesar
    if args.symbol:
        symbols = [args.symbol]
    else:
        print(f"Obteniendo top {MAX_ASSETS} activos de {args.exchange}...")
        if args.exchange == 'binance':
            symbols = get_top_assets_binance()
        else:
            symbols = get_top_assets_kucoin()
        print(f"Activos: {symbols}")

    for sym in symbols:
        print(f"\nProcesando {sym} - {args.timeframe}min...")
        try:
            if args.exchange == 'binance':
                # Binance usa símbolos sin guión, ej. SOLUSDT
                binance_sym = sym.replace('-', '')
                df_raw = fetch_klines_binance(binance_sym, f"{args.timeframe}m", args.days)
            else:
                # KuCoin usa símbolos con guión, ej. SOL-USDT
                df_raw = fetch_klines_kucoin(sym, args.timeframe, args.days)

            if df_raw.empty:
                print(f"No hay datos para {sym}")
                continue

            # Nota: Para Binance ya descargamos directamente la temporalidad solicitada.
            # Para KuCoin también descargamos la temporalidad directamente.
            # No es necesario resamplear.

            strat = StrategyHold(df_raw, initial_capital=args.initial_capital)
            trades, returns, capital_series = strat.backtest()
            report = generate_report(sym, args.timeframe, trades, returns, capital_series)

            # Guardar archivo
            safe_sym = sym.replace('/', '_').replace('-', '_')
            filename = f"reports/{safe_sym}_{args.timeframe}min.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Reporte guardado: {filename}")

        except Exception as e:
            print(f"Error procesando {sym}: {e}")

if __name__ == "__main__":
    main()
