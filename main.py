name: Generate Trading Signal

on:
  schedule:
    - cron: '55 * * * *'
  workflow_dispatch:
    inputs:
      symbol:
        description: '选择分析的币种'
        required: true
        type: choice
        options:
          - BTC
          - ETH
          - BOTH
        default: 'ETH'

jobs:
  run-strategy:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    strategy:
      matrix:
        symbol: ${{ github.event_name == 'schedule' && fromJSON('["BTC", "ETH"]') || (github.event.inputs.symbol == 'BOTH' && fromJSON('["BTC", "ETH"]') || fromJSON(format('["{0}"]', github.event.inputs.symbol))) }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run strategy analysis
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
          COINGLASS_API_KEY: ${{ secrets.COINGLASS_API_KEY }}
          DINGTALK_WEBHOOK_URL: ${{ secrets.DINGTALK_WEBHOOK_URL }}
          DINGTALK_SECRET: ${{ secrets.DINGTALK_SECRET }}
          STRATEGY_SYMBOL: ${{ matrix.symbol }}
        run: python main.py
