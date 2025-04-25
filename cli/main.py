import click

@click.command()
@click.option('--ticker', prompt='Enter NASDAQ ticker', help='The stock ticker symbol (e.g., TSLA).')
def main(ticker):
    click.echo(f"📈 분석을 시작합니다: {ticker}")
    # 여기에 수집 및 분석 로직 호출 예정

if __name__ == "__main__":
    main()