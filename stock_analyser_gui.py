import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import threading

# --- GUI 애플리케이션 클래스 ---
class StockAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("미국 주식 TOP 8 분석기")
        self.root.geometry("1400x800") # 창 크기 설정

        self.top_8_stocks_df = None # TOP 8 데이터를 저장할 변수

        # --- UI 위젯 생성 ---
        self.create_widgets()

    def create_widgets(self):
        # 1. 컨트롤 프레임 (왼쪽)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side="left", fill="y")

        # 1.1. TOP 8 조회 버튼
        self.fetch_button = ttk.Button(control_frame, text="시가총액 TOP 8 조회", command=self.start_fetch_top_8)
        self.fetch_button.pack(pady=10, fill='x')

        # 1.2. 진행 상태 표시 라벨
        self.status_label = ttk.Label(control_frame, text="버튼을 눌러 조회를 시작하세요.")
        self.status_label.pack(pady=5, fill='x')
        
        # 1.3. 구분선
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 1.4. 차트 옵션 (처음에는 비활성화)
        ttk.Label(control_frame, text="--- 차트 옵션 ---").pack()
        
        self.period_var = tk.StringVar(value='1년')
        self.chart_type_var = tk.StringVar(value='주가 변화율 (%)')

        period_options = ['1일', '5일', '1개월', '6개월', '1년', '5년', '전체']
        chart_type_options = ['주가 변화율 (%)', '절대 가격 (USD)']

        self.period_menu = ttk.OptionMenu(control_frame, self.period_var, '1년', *period_options, command=self.plot_charts)
        self.chart_type_menu = ttk.OptionMenu(control_frame, self.chart_type_var, '주가 변화율 (%)', *chart_type_options, command=self.plot_charts)
        
        self.period_menu.pack(pady=5, fill='x')
        self.chart_type_menu.pack(pady=5, fill='x')
        
        # 초기에는 비활성화 상태로 둠
        self.period_menu.config(state="disabled")
        self.chart_type_menu.config(state="disabled")

        # 2. 차트 프레임 (오른쪽)
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(side="right", fill="both", expand=True)
        
        # Matplotlib Figure와 Canvas 생성
        self.fig = plt.Figure(figsize=(28, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # --- 기능 함수 ---
    
    def start_fetch_top_8(self):
        # GUI가 멈추지 않도록 스레드를 사용하여 데이터 조회 함수를 실행
        self.fetch_button.config(state="disabled")
        self.status_label.config(text="시가총액 순위를 조회 중입니다...\n(약 1~2분 소요)")
        
        thread = threading.Thread(target=self.fetch_top_8_data)
        thread.daemon = True # 메인 프로그램 종료 시 스레드도 종료
        thread.start()

    def fetch_top_8_data(self):
        # (이전 코드의 1단계 로직과 동일)
        try:
            all_tickers = set()
            header = {'User-Agent': 'Mozilla/5.0'}
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_df = pd.read_html(requests.get(sp500_url, headers=header).text)[0]
            all_tickers.update(sp500_df['Symbol'].tolist())
            
            nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            nasdaq_df = pd.read_html(requests.get(nasdaq_url, headers=header).text)[4]
            all_tickers.update(nasdaq_df['Ticker'].tolist())
            
            stocks_with_marcap = []
            total = len(all_tickers)
            for i, symbol in enumerate(list(all_tickers)):
                self.root.after(0, lambda: self.status_label.config(text=f"종목 조회 중... ({i+1}/{total})"))
                symbol = symbol.replace('.', '-')
                stock_info = yf.Ticker(symbol).info
                market_cap, name = stock_info.get('marketCap'), stock_info.get('shortName')
                if market_cap and name:
                    stocks_with_marcap.append({'Symbol': symbol, 'Name': name, 'MarketCap': market_cap})
            
            market_cap_df = pd.DataFrame(stocks_with_marcap)
            self.top_8_stocks_df = market_cap_df.sort_values(by='MarketCap', ascending=False).head(8)
            
            # 모든 작업이 끝나면 메인 스레드에서 UI 업데이트
            self.root.after(0, self.on_fetch_complete)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("오류", f"데이터를 가져오는 중 오류가 발생했습니다:\n{e}"))
            self.root.after(0, self.reset_ui)
    
    def on_fetch_complete(self):
        messagebox.showinfo("성공", "TOP 8 기업 조회가 완료되었습니다. 이제 차트 옵션을 사용할 수 있습니다.")
        self.status_label.config(text="조회 완료!")
        self.fetch_button.config(state="normal")
        self.period_menu.config(state="normal")
        self.chart_type_menu.config(state="normal")
        self.plot_charts() # 처음 차트 그리기

    def reset_ui(self):
        self.status_label.config(text="오류 발생. 다시 시도하세요.")
        self.fetch_button.config(state="normal")

    def plot_charts(self, event=None):
        if self.top_8_stocks_df is None:
            return

        self.fig.clear() # 이전 차트 지우기
        
        period_mapping = {
            '1일': ('1d', '30m'), '5일': ('5d', '30m'), '1개월': ('1mo', '1d'),
            '6개월': ('6mo', '1d'), '1년': ('1y', '1d'), '5년': ('5y', '1wk'), '전체': ('max', '1mo')
        }
        period_name = self.period_var.get()
        chart_type = self.chart_type_var.get()
        yf_period, yf_interval = period_mapping[period_name]
        
        # 2x4 서브플롯 생성
        axes = self.fig.subplots(2, 4, sharey=(chart_type == '주가 변화율 (%)'))
        self.fig.suptitle(f'TOP 8 기업 {period_name} 주가 변화 비교 ({chart_type})', fontsize=16)

        for ax, (index, row) in zip(axes.flatten(), self.top_8_stocks_df.iterrows()):
            ticker, name = row['Symbol'], row['Name']
            data = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False)
            
            if data.empty:
                ax.set_title(f"{name}\n(데이터 없음)")
                continue
            
            if chart_type == '주가 변화율 (%)':
                normalized_price = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                ax.plot(data.index, normalized_price)
                ax.set_ylabel("변화율 (%)")
                ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            else:
                ax.plot(data.index, data['Close'])
                ax.set_ylabel("가격 (USD)")
            
            ax.set_title(f"{name} ({ticker})")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()


# --- 메인 프로그램 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzerApp(root)
    root.mainloop()
