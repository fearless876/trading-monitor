#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号监控系统 - 多币种量化监控与提醒工具
集成技术指标分析、K线形态识别、交易量确认、市场情绪评估
优化版：推送内容更加简洁专业
"""

import os
import time
import requests
import json
import base64
import hmac
import hashlib
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timezone, timedelta
import threading
import schedule
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_monitor.log'),
        logging.StreamHandler()
    ]
)

class TradingSignalMonitor:
    def __init__(self):
        # 从环境变量获取API配置
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.webhook_url = os.getenv('WECHAT_WEBHOOK_URL')
        
        # 检查必要的环境变量
        if not all([self.api_key, self.secret_key, self.passphrase, self.webhook_url]):
            raise ValueError("请设置必要的环境变量：OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, WECHAT_WEBHOOK_URL")
        
        self.base_url = "https://www.okx.com"
        
        # 监控币种列表
        self.symbols = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT', 'DOGE-USDT', 'LTC-USDT']
        
        # 时间周期权重 (数值越大权重越高)
        self.timeframe_weights = {
            '1w': 4,    # 周线
            '1D': 3,    # 日线
            '12H': 2,   # 12小时
            '15m': 1    # 15分钟
        }
        
        # 市场情绪缓存
        self.market_sentiment = "中性"
        self.sentiment_score = 0
        
        logging.info("交易信号监控系统初始化完成")
    
    def generate_signature(self, timestamp, method, request_path, body=""):
        """生成OKX API签名"""
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()
    
    def get_timestamp(self):
        """获取ISO 8601格式时间戳"""
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z")
    
    def send_request(self, method, endpoint, body=None):
        """发送API请求"""
        url = self.base_url + endpoint
        timestamp = self.get_timestamp()
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": self.generate_signature(timestamp, method, endpoint, json.dumps(body) if body else ""),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(method, url, headers=headers, json=body, timeout=10)
            return response.json()
        except Exception as e:
            logging.error(f"API请求失败: {e}")
            return None
    
    def get_kline_data(self, symbol, timeframe, limit=200):
        """获取K线数据"""
        endpoint = f"/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
        response = self.send_request("GET", endpoint)
        
        if response and response.get('code') == '0':
            data = response['data']
            # 转换为DataFrame格式
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
            # 转换数据类型
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        return None
    
    def calculate_support_resistance(self, df, window=20):
        """计算支撑位和阻力位"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # 寻找局部高点和低点
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
        
        # 计算Fibonacci回撤位
        if len(df) >= 50:
            high_price = df['high'].iloc[-50:].max()
            low_price = df['low'].iloc[-50:].min()
            diff = high_price - low_price
            
            fib_levels = {
                '23.6%': high_price - 0.236 * diff,
                '38.2%': high_price - 0.382 * diff,
                '50%': high_price - 0.5 * diff,
                '61.8%': high_price - 0.618 * diff
            }
        else:
            fib_levels = {}
        
        # 返回最近的支撑阻力位
        current_price = df['close'].iloc[-1]
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        return {
            'support': nearest_support,
            'resistance': nearest_resistance,
            'fibonacci': fib_levels
        }
    
    def identify_candlestick_patterns(self, df):
        """识别K线形态"""
        if len(df) < 5:
            return []
        
        patterns = []
        
        # 使用talib识别K线形态
        try:
            # 锤子线（蜻蜓点水）
            hammer = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            if hammer.iloc[-1] > 0:
                patterns.append("锤子线")
            
            # 倒锤子线
            inverted_hammer = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
            if inverted_hammer.iloc[-1] > 0:
                patterns.append("倒锤子线")
            
            # 射击之星
            shooting_star = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            if shooting_star.iloc[-1] > 0:
                patterns.append("射击之星")
            
            # 上吊线
            hanging_man = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            if hanging_man.iloc[-1] > 0:
                patterns.append("上吊线")
            
            # 吞没形态
            engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            if engulfing.iloc[-1] > 0:
                patterns.append("看涨吞没")
            elif engulfing.iloc[-1] < 0:
                patterns.append("看跌吞没")
            
            # 早晨之星
            morning_star = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            if morning_star.iloc[-1] > 0:
                patterns.append("早晨之星")
            
            # 黄昏之星
            evening_star = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            if evening_star.iloc[-1] > 0:
                patterns.append("黄昏之星")
            
            # 十字星
            doji = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            if doji.iloc[-1] > 0:
                patterns.append("十字星")
                
        except Exception as e:
            logging.warning(f"K线形态识别失败: {e}")
        
        return patterns
    
    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        if len(df) < 50:
            return None
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line
        
        # RSI
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        
        # 成交量平均值
        volume_avg = df['volume'].rolling(window=20).mean()
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        return {
            'macd_line': macd_line.iloc[-1],
            'macd_signal': signal_line.iloc[-1],
            'macd_histogram': macd_histogram.iloc[-1],
            'macd_hist_prev': macd_histogram.iloc[-2],
            'rsi': rsi[-1],
            'volume_current': df['volume'].iloc[-1],
            'volume_avg': volume_avg.iloc[-1],
            'price': df['close'].iloc[-1],
            'bb_upper': bb_upper[-1],
            'bb_middle': bb_middle[-1],
            'bb_lower': bb_lower[-1]
        }
    
    def analyze_signal_strength(self, indicators, patterns, sr_levels, timeframe):
        """分析信号强度"""
        signals = []
        score = 0
        signal_type = "中性"
        
        # MACD信号分析
        macd_signal = ""
        if indicators['macd_line'] > indicators['macd_signal'] and indicators['macd_histogram'] > indicators['macd_hist_prev']:
            if indicators['macd_histogram'] > 0:
                macd_signal = "MACD金叉确认"
                score += 2
                signal_type = "看涨"
            else:
                macd_signal = "MACD向上趋势"
                score += 1
                signal_type = "偏多"
        elif indicators['macd_line'] < indicators['macd_signal'] and indicators['macd_histogram'] < indicators['macd_hist_prev']:
            if indicators['macd_histogram'] < 0:
                macd_signal = "MACD死叉确认"
                score -= 2
                signal_type = "看跌"
            else:
                macd_signal = "MACD向下趋势"
                score -= 1
                signal_type = "偏空"
        
        # RSI信号分析
        rsi_signal = ""
        if indicators['rsi'] > 70:
            rsi_signal = f"RSI超买({indicators['rsi']:.1f})"
            score -= 1
        elif indicators['rsi'] < 30:
            rsi_signal = f"RSI超卖({indicators['rsi']:.1f})"
            score += 1
        elif 30 <= indicators['rsi'] <= 70:
            rsi_signal = f"RSI正常区间({indicators['rsi']:.1f})"
        
        # 布林带分析
        bb_signal = ""
        current_price = indicators['price']
        if current_price > indicators['bb_upper']:
            bb_signal = "价格触及布林带上轨"
            score -= 0.5
        elif current_price < indicators['bb_lower']:
            bb_signal = "价格触及布林带下轨"
            score += 0.5
        else:
            bb_position = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            if bb_position > 0.7:
                bb_signal = "价格接近布林带上轨"
            elif bb_position < 0.3:
                bb_signal = "价格接近布林带下轨"
            else:
                bb_signal = "价格在布林带中轨附近"
        
        # 成交量确认
        volume_confirmation = ""
        if indicators['volume_current'] > indicators['volume_avg'] * 1.5:
            volume_confirmation = "成交量显著放大"
            score += 1
        elif indicators['volume_current'] < indicators['volume_avg'] * 0.7:
            volume_confirmation = "成交量萎缩"
            score -= 0.5
        else:
            volume_confirmation = "成交量正常"
        
        # K线形态评分
        pattern_signal = ""
        if patterns:
            pattern_signal = "，".join(patterns)
            # 根据形态类型调整评分
            for pattern in patterns:
                if any(word in pattern for word in ["看涨", "早晨之星", "锤子"]):
                    score += 1
                elif any(word in pattern for word in ["看跌", "黄昏之星", "射击"]):
                    score -= 1
        
        # 支撑阻力位分析
        sr_analysis = ""
        if sr_levels['resistance'] and abs(current_price - sr_levels['resistance']) / current_price < 0.02:
            sr_analysis = f"接近阻力位{sr_levels['resistance']:.4f}"
            if signal_type == "看涨":
                sr_analysis += "，突破后可能延续上涨"
        if sr_levels['support'] and abs(current_price - sr_levels['support']) / current_price < 0.02:
            sr_analysis = f"接近支撑位{sr_levels['support']:.4f}"
            if signal_type == "看跌":
                sr_analysis += "，跌破后可能加速下跌"
        
        # 时间周期权重调整
        score *= self.timeframe_weights.get(timeframe, 1)
        
        # 评级
        if abs(score) >= 6:
            rating = "高"
        elif abs(score) >= 4:
            rating = "中高"
        elif abs(score) >= 2:
            rating = "中"
        else:
            rating = "低"
        
        return {
            'signal_type': signal_type,
            'score': score,
            'rating': rating,
            'macd_signal': macd_signal,
            'rsi_signal': rsi_signal,
            'bb_signal': bb_signal,
            'volume_confirmation': volume_confirmation,
            'pattern_signal': pattern_signal,
            'sr_analysis': sr_analysis
        }
    
    def update_market_sentiment(self):
        """更新市场情绪"""
        try:
            btc_data = self.get_kline_data('BTC-USDT', '1D', 10)
            eth_data = self.get_kline_data('ETH-USDT', '1D', 10)
            
            if btc_data is None or eth_data is None:
                return
            
            # 计算BTC和ETH的涨跌幅
            btc_change = (btc_data['close'].iloc[-1] - btc_data['close'].iloc[-2]) / btc_data['close'].iloc[-2] * 100
            eth_change = (eth_data['close'].iloc[-1] - eth_data['close'].iloc[-2]) / eth_data['close'].iloc[-2] * 100
            
            # 计算整体市场情绪得分
            sentiment_score = (btc_change + eth_change) / 2
            
            if sentiment_score > 3:
                self.market_sentiment = "乐观"
                self.sentiment_score = sentiment_score
            elif sentiment_score < -3:
                self.market_sentiment = "悲观"
                self.sentiment_score = sentiment_score
            else:
                self.market_sentiment = "中性"
                self.sentiment_score = sentiment_score
                
        except Exception as e:
            logging.error(f"更新市场情绪失败: {e}")
    
    def send_wechat_message(self, message):
        """发送企业微信消息"""
        try:
            data = {
                "msgtype": "text",
                "text": {"content": message}
            }
            response = requests.post(self.webhook_url, json=data, timeout=10)
            if response.status_code == 200:
                logging.info("消息发送成功")
            else:
                logging.error(f"消息发送失败: {response.text}")
        except Exception as e:
            logging.error(f"发送微信消息失败: {e}")
    
    def format_signal_message(self, symbol, timeframe, analysis, indicators, patterns, sr_levels):
        """格式化信号消息"""
        current_price = indicators['price']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"""交易信号提醒
币种: {symbol}
时间周期: {timeframe}
当前价格: {current_price:.6f}
信号时间: {timestamp}
信号强度: {analysis['rating']}
信号方向: {analysis['signal_type']}

技术指标分析:
MACD: {analysis['macd_signal']}
RSI: {analysis['rsi_signal']}
布林带: {analysis['bb_signal']}
成交量: {analysis['volume_confirmation']}"""

        if analysis['pattern_signal']:
            message += f"\nK线形态: {analysis['pattern_signal']}"
        
        if analysis['sr_analysis']:
            message += f"\n关键位分析: {analysis['sr_analysis']}"
        
        message += f"\n\n市场情绪: {self.market_sentiment} ({self.sentiment_score:.1f}%)"
        
        # 操作建议
        if analysis['signal_type'] == "看涨":
            suggestion = "建议关注做多机会，设置合理止损位。"
        elif analysis['signal_type'] == "看跌":
            suggestion = "建议关注做空机会，注意反弹风险。"
        elif analysis['signal_type'] == "偏多":
            suggestion = "市场偏向多头，可适度关注上涨机会。"
        elif analysis['signal_type'] == "偏空":
            suggestion = "市场偏向空头，建议减仓或观望。"
        else:
            suggestion = "当前信号不明确，建议观望。"
        
        message += f"\n操作建议: {suggestion}"
        
        return message
    
    def monitor_symbol(self, symbol):
        """监控单个币种"""
        logging.info(f"开始监控 {symbol}")
        
        for timeframe in ['15m', '12H', '1D', '1w']:
            try:
                df = self.get_kline_data(symbol, timeframe)
                if df is None or len(df) < 50:
                    continue
                
                indicators = self.calculate_technical_indicators(df)
                if indicators is None:
                    continue
                
                patterns = self.identify_candlestick_patterns(df)
                sr_levels = self.calculate_support_resistance(df)
                analysis = self.analyze_signal_strength(indicators, patterns, sr_levels, timeframe)
                
                # 只推送有效信号（评分绝对值大于等于2）
                if abs(analysis['score']) >= 2 and analysis['rating'] != "低":
                    message = self.format_signal_message(symbol, timeframe, analysis, indicators, patterns, sr_levels)
                    self.send_wechat_message(message)
                    time.sleep(1)  # 避免频繁推送
                
            except Exception as e:
                logging.error(f"监控 {symbol} {timeframe} 失败: {e}")
    
    def run_full_monitor(self):
        """运行完整监控"""
        logging.info("开始执行全市场监控...")
        self.update_market_sentiment()
        
        for symbol in self.symbols:
            self.monitor_symbol(symbol)
            time.sleep(2)  # 避免API频率限制
        
        logging.info("全市场监控完成")
    
    def generate_daily_report(self):
        """生成每日晨报"""
        logging.info("生成每日晨报...")
        
        report = f"""每日市场晨报
报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
市场情绪: {self.market_sentiment} ({self.sentiment_score:.1f}%)

主要币种日线分析:"""
        
        signals_summary = {"看涨": 0, "看跌": 0, "中性": 0}
        
        for symbol in self.symbols:
            try:
                df = self.get_kline_data(symbol, '1D')
                if df is None or len(df) < 50:
                    continue
                
                indicators = self.calculate_technical_indicators(df)
                if indicators is None:
                    continue
                
                patterns = self.identify_candlestick_patterns(df)
                sr_levels = self.calculate_support_resistance(df)
                analysis = self.analyze_signal_strength(indicators, patterns, sr_levels, '1D')
                
                # 统计信号分布
                if "看涨" in analysis['signal_type']:
                    signals_summary["看涨"] += 1
                elif "看跌" in analysis['signal_type']:
                    signals_summary["看跌"] += 1
                else:
                    signals_summary["中性"] += 1
                
                # 只报告重要信号
                if abs(analysis['score']) >= 3:
                    report += f"\n{symbol}: {analysis['signal_type']} ({analysis['rating']})"
                    if analysis['macd_signal']:
                        report += f" - {analysis['macd_signal']}"
                
            except Exception as e:
                logging.error(f"生成晨报时分析 {symbol} 失败: {e}")
        
        report += f"\n\n信号统计: 看涨{signals_summary['看涨']}个, 看跌{signals_summary['看跌']}个, 中性{signals_summary['中性']}个"
        
        # 市场建议
        if signals_summary["看涨"] > signals_summary["看跌"]:
            market_advice = "整体偏向多头，可关注优质币种的上涨机会。"
        elif signals_summary["看跌"] > signals_summary["看涨"]:
            market_advice = "整体偏向空头，建议控制仓位，注意风险。"
        else:
            market_advice = "市场方向不明确，建议观望等待更清晰信号。"
        
        report += f"\n今日操作建议: {market_advice}"
        
        self.send_wechat_message(report)
        logging.info("每日晨报发送完成")
    
    def start_monitoring(self):
        """启动监控系统"""
        logging.info("启动交易信号监控系统...")
        
        # 设置定时任务
        schedule.every(15).minutes.do(self.run_full_monitor)  # 每15分钟全市场扫描
        schedule.every().day.at("09:00").do(self.generate_daily_report)  # 每日9点晨报
        
        # 立即执行一次
        self.run_full_monitor()
        
        # 持续运行
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                logging.info("监控系统停止")
                break
            except Exception as e:
                logging.error(f"监控系统错误: {e}")
                time.sleep(60)

def main():
    """主函数"""
    try:
        monitor = TradingSignalMonitor()
        monitor.start_monitoring()
    except Exception as e:
        logging.error(f"系统启动失败: {e}")

if __name__ == "__main__":
    main()