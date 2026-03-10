#!/usr/bin/env bash

ollama serve &

sleep 10

ollama pull llama3

streamlit run app.py --server.port $PORT --server.address 0.0.0.0