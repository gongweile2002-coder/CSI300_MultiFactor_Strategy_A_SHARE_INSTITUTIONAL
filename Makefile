
.PHONY: test demo ultimate-demo check doctor prepare dashboard

test:
	python -m pytest -q

demo:
	python live.py demo

ultimate-demo:
	python run_demo_v6.py
	python run_ultimate_demo.py

check:
	python production_check.py

dashboard:
	streamlit run dashboard_app.py

doctor:
	python live.py doctor

prepare:
	python live.py prepare
