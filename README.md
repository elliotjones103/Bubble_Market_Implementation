### READ ME ###
### Delta-band Bubble Detector ###

This repository contains a small Python package and demo app that estimate the martingale defect from option prices, based on the delta-band estimator developed in my MSc thesis: Modelling and Quantifying the Martingale Defect in Asset Price Bubbles.
The code can work with both simulated models, (Black–Scholes, CEV strict local martingale, SABR $\beta$=1) and real option chains (e.g. AAPL, NVDA, INTC) downloaded from Yahoo Finance. 
A Streamlit dashboard lets you interactively pick a ticker, expiry and delta band and visualise the reconstructed fundamental call curve and inferred bubble component.
